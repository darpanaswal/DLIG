"""
DLIG (Diffusion Language Integrated Gradients) attribution implementation.
"""

import torch
import traceback
from contextlib import contextmanager

class DLIGAttribution:
    def __init__(self, model, tokenizer, hook_manager, integration_steps=20):
        self.model = model
        self.tokenizer = tokenizer
        self.integration_steps = integration_steps
        self.hook_manager = hook_manager
        
        # DLIG attributes
        self.dlig_scores = []
        self.original_input_length = None
        self.relevant_token_indices = []

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
                     # Attempt generic reshaping if dimensions match but shapes differ slightly
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
    def activation_intervention(self, layer_name=None, interpolated=None):
        """Context manager to manage activation intervention."""
        # Fix: passing None allows the manager to use the already-resolved layer
        hooked_layer = self.hook_manager._get_layer(layer_name)
        self.use_interpolated_activations = interpolated is not None
        self.interpolated_activations = interpolated
        
        hook_handle = hooked_layer.register_forward_hook(self._intervention_hook_fn)
        try:
            yield
        finally:
            hook_handle.remove()
            self.use_interpolated_activations = False
            self.interpolated_activations = None

    def compute_dlig_at_timestep(self, step, x_t, logits, mask_token_id, original_length, attention_mask):
        """
        Computes DLIG for a specific diffusion timestep t[cite: 17, 20].
        """
        self.model.eval()
        
        if attention_mask.dtype == torch.long:
            attention_mask = attention_mask.float()
        
        with torch.no_grad():
            # a = h_t(X_t, C) [cite: 21, 25]
            real_act = self.get_layer_activations(x_t, attention_mask)
            
            # a' = h_t(X_t, ∅) [cite: 22, 25]
            baseline_inp = self.create_baseline_input(x_t, mask_token_id, original_length)
            baseline_act = self.get_layer_activations(baseline_inp, attention_mask)

        activation_diff = real_act - baseline_act
        accumulated_gradients = torch.zeros_like(real_act)

        # Riemann Sum Approximation [cite: 14, 26, 27]
        for k in range(1, self.integration_steps + 1):
            alpha = k / self.integration_steps
            interpolated = baseline_act + alpha * activation_diff
            interpolated = interpolated.detach().requires_grad_(True)

            with self.activation_intervention(self.hook_manager.layer_name, interpolated):
                outputs = self.model(input_ids=x_t, attention_mask=attention_mask)
                # Compute F_t for the current interpolated path [cite: 25, 27]
                target_score = self._compute_target_score(outputs, x_t, original_length)

            if interpolated.grad is not None:
                interpolated.grad.zero_()
            
            gradients = torch.autograd.grad(target_score, interpolated, retain_graph=False)[0]
            accumulated_gradients += gradients.detach()

        avg_gradients = accumulated_gradients / self.integration_steps
        dlig_raw = activation_diff.detach() * avg_gradients

        return self._process_results(dlig_raw.cpu(), original_length, step, activation_diff)

    def _prepare_attention_mask(self, attention_mask):
        """Ensure attention_mask has the correct dtype for the model."""
        if attention_mask is None:
            return None
        
        # Convert long/int to float if needed
        if attention_mask.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            attention_mask = attention_mask.float()
        
        return attention_mask

    def get_layer_activations(self, input_ids, attention_mask):
        """Helper to run a forward pass and capture specific layer output."""
        attention_mask = self._prepare_attention_mask(attention_mask)
        
        layer_acts = []
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                hidden_states = out[0]
            else:
                hidden_states = out
            layer_acts.append(hidden_states.detach().clone())

        # Fix: No longer passing a string here; uses the stored module from register_hook
        layer = self.hook_manager._get_layer()
        handle = layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()

        if not layer_acts:
            raise RuntimeError(f"Failed to capture activations at layer {layer}")
            
        return layer_acts[0]

    def create_baseline_input(self, x_t, mask_token_id, original_length):
        """
        Creates baseline input for DLIG.
        
        CRITICAL FIX: The baseline should have:
        - SAME noisy/generated tokens (X_t remains unchanged)
        - NULL condition (mask the prompt/condition part)
        
        Math: h_t(X_t, ∅) means same noise state, no conditioning.
        """
        baseline = x_t.clone()
        
        # Mask ONLY the prompt tokens (the condition C)
        # This creates the "null condition" while preserving the noise state
        baseline[:, :original_length] = mask_token_id
        
        return baseline
    
    def _compute_target_score(self, outputs, input_ids, original_length):
        """
        Computes F_t: The score of the PREDICTED FINAL SEQUENCE x̂_0.
        Uses the Top-1 log-probability to represent the model's current 
        internal prediction confidence.
        """
        logits = outputs.logits
        
        # Focus on the generated portion (after the prompt/condition C) [cite: 21, 45]
        gen_logits = logits[:, original_length:, :] 
        
        # Dynamic Scoring: Sum of max log-probs for the predicted tokens
        # This represents "how certain is the model about its current x̂_0 prediction"
        log_probs = torch.log_softmax(gen_logits, dim=-1)
        max_log_probs, _ = log_probs.max(dim=-1)
        return max_log_probs.sum()

    def _process_results(self, dlig_raw, original_length, step, activation_diff):
        """Process raw DLIG results into final scores."""
        # Focus on the prompt/condition section (C)
        if self.relevant_token_indices:
            input_dlig = dlig_raw[:, self.relevant_token_indices, :].detach().cpu()
        else:
            input_dlig = dlig_raw[:, :original_length, :].detach().cpu()
            
        token_scores = input_dlig.sum(dim=-1)  # Sum across hidden dimension
        
        return {
            'step': step,
            'token_scores': token_scores,
            'full_dlig': input_dlig,
            'activation_diff_norm': activation_diff[:, :original_length, :].norm(dim=-1).detach().cpu()
        }

    def identify_relevant_tokens(self, input_tokens):
        """
        Identify relevant tokens (non-special tokens) for attribution.
        This filters out padding, BOS, EOS, etc.
        """
        special_tokens = {
            self.tokenizer.pad_token,
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
            self.tokenizer.unk_token,
            self.tokenizer.sep_token,
            self.tokenizer.cls_token,
            self.tokenizer.mask_token,
        }
        # Remove None values
        special_tokens = {t for t in special_tokens if t is not None}
        
        relevant_indices = []
        for idx, token in enumerate(input_tokens):
            if token not in special_tokens:
                relevant_indices.append(idx)
        
        return relevant_indices

    def set_target_output(self, target_text):
        """
        Set a specific target output for attribution.
        """
        target_ids = self.tokenizer.encode(target_text, return_tensors="pt")
        self.target_output_ids = target_ids.to(next(self.model.parameters()).device)
        print(f"[DEBUG] Target output set: {target_text}")
        print(f"[DEBUG] Target IDs shape: {self.target_output_ids.shape}")

    def generation_logits_hook_func(self, step, x, logits):
        """
        Hook called during generation loop.
        x: The current input_ids at diffusion step t.
        """
        if step is not None and self.original_input_length is not None:
            mask_token_id = (
                self.tokenizer.mask_token_id 
                if self.tokenizer.mask_token_id is not None 
                else self.tokenizer.pad_token_id
            )
            
            # Create attention_mask with proper dtype (float instead of long)
            attention_mask = torch.ones_like(x, dtype=torch.float32, device=x.device)

            try:
                dlig_result = self.compute_dlig_at_timestep(
                    step, x, logits, mask_token_id, self.original_input_length, attention_mask
                )
                self.dlig_scores.append(dlig_result)
                print(f"Step {step}: DLIG computed successfully.")
            except Exception as e:
                print(f"Step {step}: DLIG failed - {e}")
                traceback.print_exc()

            # Optional: Show intermediate generation
            try:
                decoded = self.tokenizer.decode(x[0, self.original_input_length:], skip_special_tokens=True)
                print(f"Step {step} partial generation: {decoded[:50]}...")
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
        self.target_output_ids = None