"""
DLIG (Diffusion Language Integrated Gradients) attribution implementation.
"""

import torch
import traceback
from config import INTEGRATION_STEPS
from contextlib import contextmanager

class DLIGAttribution:
    def __init__(self, model, tokenizer, hook_manager, integration_steps=INTEGRATION_STEPS):
        self.model = model
        self.tokenizer = tokenizer
        self.integration_steps = integration_steps
        self.hook_manager = hook_manager
        
        # DLIG attributes
        self.dlig_scores = []
        self.original_input_length = None
        self.relevant_token_indices = []
        self.target_output_ids = None

        # For activation manipulation
        self.interpolated_activations = None
        self.use_interpolated_activations = False
    
    def _intervention_hook_fn(self, module, inputs, outputs):
        """
        Replaces layer output with interpolated activations during the Riemann sum loop.
        """
        if self.use_interpolated_activations and self.interpolated_activations is not None:
            # Handle Tuple: transformer layers commonly return tuples (hidden_states, attentions, etc.)
            if isinstance(outputs, tuple):
                ref_tensor = outputs[0]
                if self.interpolated_activations.shape != ref_tensor.shape:
                     # Attempt generic reshaping if dimensions match but shapes differ slightly (e.g. 1, S, H vs S, H)
                    if self.interpolated_activations.numel() == ref_tensor.numel():
                         self.interpolated_activations = self.interpolated_activations.view_as(ref_tensor)
                    else:
                        raise ValueError(
                            f"Shape mismatch in intervention_hook: expected {ref_tensor.shape}, "
                            f"got {self.interpolated_activations.shape}"
                        )
                # Return all outputs, replacing only hidden_states
                return (self.interpolated_activations,) + outputs[1:]
            else:
                if self.interpolated_activations.shape != outputs.shape:
                    raise ValueError(
                        f"Shape mismatch: expected {outputs.shape}, got {self.interpolated_activations.shape}"
                    )
                return self.interpolated_activations
        return outputs

    @contextmanager
    def activation_intervention(self, layer_name, interpolated=None):
        """Context manager to manage activation intervention."""
        hooked_layer = self.hook_manager._get_layer(layer_name)
        self.use_interpolated_activations = interpolated is not None
        self.interpolated_activations = interpolated
        
        # Register the intervention hook
        hook_handle = hooked_layer.register_forward_hook(self._intervention_hook_fn)
        try:
            yield
        finally:
            hook_handle.remove()
            self.use_interpolated_activations = False
            self.interpolated_activations = None

    def compute_dlig_at_timestep(self, step, x_t, logits, mask_token_id, original_length, attention_mask):
        """
        Computes DLIG for a specific diffusion timestep t.
        Math Reference: DLIG_t(a) formula[cite: 29].
        """
        # Ensure model gradients are enabled for the backward pass later
        self.model.eval()
        self.model.zero_grad()
        
        # 1. Capture Real Activations: a = h_t(X_t, C) [cite: 24]
        with torch.no_grad():
            real_act = self.get_layer_activations(x_t, attention_mask)
            
            # 2. Create Baseline Input: X_t with null condition 
            # CRITICAL FIX: Do not cache. Create baseline from CURRENT x_t
            baseline_inp = self.create_baseline_input(x_t, mask_token_id, original_length)
            
            # 3. Capture Baseline Activations: a' = h_t(X_t, empty) 
            baseline_act = self.get_layer_activations(baseline_inp, attention_mask)

        # 4. Define Path: gamma(alpha) = a' + alpha(a - a') [cite: 27]
        activation_diff = real_act - baseline_act
        accumulated_gradients = torch.zeros_like(real_act)

        # 5. Riemann Sum Approximation [cite: 29]
        for k in range(1, self.integration_steps + 1):
            alpha = k / self.integration_steps
            
            # Interpolate activations
            interpolated = baseline_act + alpha * activation_diff
            interpolated.requires_grad_(True)

            # Inject interpolated activations into the model
            with self.activation_intervention(self.hook_manager.layer_name, interpolated):
                # We need gradients here
                outputs = self.model(input_ids=x_t, attention_mask=attention_mask)
                
                # Calculate Score F_t 
                # F_t is the score of the predicted final sequence
                target_score = self._compute_target_score(outputs.logits, x_t, original_length)

            # Compute gradients: nabla_a F_t [cite: 29]
            gradients = torch.autograd.grad(target_score, interpolated)[0]
            accumulated_gradients += gradients.detach()

        # 6. Final Calculation: (a - a') * Avg(Gradients)
        avg_gradients = accumulated_gradients / self.integration_steps
        dlig_raw = activation_diff * avg_gradients

        return self._process_results(dlig_raw.cpu(), original_length, step, activation_diff)

    def get_layer_activations(self, input_ids, attention_mask):
        """Helper to run a forward pass and capture specific layer output."""
        layer_acts = []
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                hidden_states = out[0]
            else:
                hidden_states = out
            layer_acts.append(hidden_states.detach().clone())

        layer = self.hook_manager._get_layer()
        # Register temporary hook
        handle = layer.register_forward_hook(hook_fn)
        
        try:
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()

        if not layer_acts:
            raise RuntimeError(f"Failed to capture activations at layer {self.hook_manager.layer_name}")
            
        return layer_acts[0]

    def create_baseline_input(self, x_t, mask_token_id, original_length):
        """
        Creates baseline input for the CURRENT timestep.
        Math: X_t is the same (noise is preserved), but condition C is nullified.
        """
        baseline = x_t.clone()
        
        # Mask the prompt tokens (0 to original_length)
        # This effectively creates h(X_t, empty_set)
        baseline[:, :original_length] = mask_token_id
        
        return baseline
    
    def _compute_target_score(self, logits, input_ids, original_length):
        """
        Computes F_t: The score of the target sequence.
        If a target is set, use that. If not, use the model's own predicted probabilities 
        for the generated part (entropy minimization / confidence).
        """
        # We only care about the score of the generated tokens (after original_length)
        gen_logits = logits[:, original_length:, :] # [Batch, Gen_Seq_Len, Vocab]
        
        if self.target_output_ids is not None:
            # If we have a ground truth target, sum log-probs of that target
            # Align target length with current generation length if necessary
            target_len = min(gen_logits.shape[1], self.target_output_ids.shape[1])
            relevant_logits = gen_logits[:, :target_len, :]
            relevant_targets = self.target_output_ids[:, :target_len]
            
            # Gather log probs of target tokens
            log_probs = torch.log_softmax(relevant_logits, dim=-1)
            target_log_probs = torch.gather(log_probs, 2, relevant_targets.unsqueeze(-1)).squeeze(-1)
            return target_log_probs.sum()
        else:
            # If no target, maximize the confidence of the model's OWN prediction (Top-1)
            # This represents "how confident is the model in this specific outcome"
            probs = torch.softmax(gen_logits, dim=-1)
            max_probs, _ = probs.max(dim=-1)
            return torch.log(max_probs).sum()

    def _process_results(self, dlig_raw, original_length, step, activation_diff):
        """Process raw DLIG results into final scores."""
        # Focus on the user input section (Condition C)
        if self.relevant_token_indices:
            input_dlig = dlig_raw[:, self.relevant_token_indices, :].detach().cpu()
        else:
            input_dlig = dlig_raw[:, :original_length, :].detach().cpu()
            
        token_scores = input_dlig.sum(dim=-1) # Sum across hidden dimension
        
        return {
            'step': step,
            'token_scores': token_scores,
            'full_dlig': input_dlig,
            'activation_diff_norm': activation_diff[:, :original_length, :].norm(dim=-1).detach().cpu()
        }

    # ... [Keep set_target_output, identify_relevant_tokens, etc. as they were] ...

    def generation_logits_hook_func(self, step, x, logits):
        """
        Hook called during generation loop.
        x: The current latent/input input_ids at step t.
        """
        if step is not None and self.original_input_length is not None:
            # Use pad_token_id or mask_token_id for the "Null" condition
            mask_token_id = (
                self.tokenizer.mask_token_id 
                if self.tokenizer.mask_token_id is not None 
                else self.tokenizer.pad_token_id
            )
            
            attention_mask = torch.ones_like(x, dtype=torch.long, device=x.device)

            # Compute DLIG for this step
            try:
                dlig_result = self.compute_dlig_at_timestep(
                    step, x, logits, mask_token_id, self.original_input_length, attention_mask
                )
                self.dlig_scores.append(dlig_result)
                print(f"Step {step}: DLIG computed.")
            except Exception as e:
                print(f"Step {step}: DLIG failed - {e}")
                traceback.print_exc()

            # Optional: Intermediate decoding
            try:
                decoded = self.tokenizer.decode(x[0, self.original_input_length:], skip_special_tokens=True)
                print(f"Step {step} generation state: {decoded[:50]}...")
            except: 
                pass

        return logits

    def set_original_input_length(self, length):
        self.original_input_length = length

    def get_relevant_token_indices(self):
        return self.relevant_token_indices

    def set_relevant_token_indices(self, input_tokens):
        self.relevant_token_indices = self.identify_relevant_tokens(input_tokens)
        relevant_tokens = [input_tokens[idx] for idx in self.relevant_token_indices]
        print(f"[DEBUG] Relevant token indices: {self.relevant_token_indices}")
        print(f"[DEBUG] Relevant tokens: {relevant_tokens}")

    def get_dlig_scores(self):
        return self.dlig_scores

    def reset_scores(self):
        self.dlig_scores = []
        self.relevant_token_indices = []