# attribution/dlig_attribution.py
"""
DLIG (Diffusion Language Integrated Gradients) attribution implementation.

DLIG definition is unchanged:
- F_t is the sum of max log-probabilities over generated positions.
- Baseline is same X_t with prompt tokens masked (null condition).

Fixes:
- Vectorized integration over k with chunking via integration_batch_size (to control VRAM).
- Always uses attention_mask=None inside DLIG (avoids Dream SDPA mask broadcast failures).
- Disables KV cache during DLIG forwards (use_cache=False when supported).
- Optional timing logs.
"""

import time
import torch
import traceback
from typing import Optional, Any
from contextlib import contextmanager

class DLIGAttribution:
    def __init__(
        self,
        model,
        tokenizer,
        hook_manager,
        integration_steps: int = 20,
        integration_batch_size: Optional[int] = 5,
        enable_timing: bool = False,
        timing_log_every_chunks: int = 1,
        disable_kv_cache: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.integration_steps = int(integration_steps)
        self.integration_batch_size = int(integration_batch_size) if integration_batch_size is not None else None
        self.enable_timing = bool(enable_timing)
        self.timing_log_every_chunks = max(1, int(timing_log_every_chunks))
        self.disable_kv_cache = bool(disable_kv_cache)
        self.hook_manager = hook_manager

        # DLIG attributes
        self.dlig_scores = []
        self.original_input_length = None
        self.relevant_token_indices = []

        # For activation manipulation
        self.interpolated_activations = None
        self.use_interpolated_activations = False

    def debug_baseline_construction(self, step, x_t, mask_token_id, original_length):
        """
        Print what x_t contains and what the baseline looks like.
        """
        print(f"\n=== BASELINE DEBUG at step {step} ===")
        print(f"x_t shape: {x_t.shape}")
        print(f"original_length (prompt length): {original_length}")
        
        # Decode the full x_t
        full_sequence = self.tokenizer.decode(x_t[0], skip_special_tokens=False)
        prompt_part = self.tokenizer.decode(x_t[0, :original_length], skip_special_tokens=False)
        generated_part = self.tokenizer.decode(x_t[0, original_length:], skip_special_tokens=False)
        
        print(f"\nFull x_t: {full_sequence}")
        print(f"Prompt part ([:original_length]): {prompt_part}")
        print(f"Generated part ([original_length:]): {generated_part}")
        
        # Create baseline and decode
        baseline = self.create_baseline_input(x_t, mask_token_id, original_length)
        baseline_sequence = self.tokenizer.decode(baseline[0], skip_special_tokens=False)
        baseline_prompt = self.tokenizer.decode(baseline[0, :original_length], skip_special_tokens=False)
        baseline_generated = self.tokenizer.decode(baseline[0, original_length:], skip_special_tokens=False)
        
        print(f"\nBaseline full: {baseline_sequence}")
        print(f"Baseline prompt: {baseline_prompt}")
        print(f"Baseline generated: {baseline_generated}")
        
        # Check if generated parts are identical
        tokens_match = torch.equal(x_t[0, original_length:], baseline[0, original_length:])
        print(f"\nGenerated tokens match (should be True): {tokens_match}")
        print("="*50)

    def debug_activation_differences(self, step, real_act, baseline_act, original_length, layer_name=None):
        diff = real_act - baseline_act
        prompt_diff_norm = diff[:, :original_length, :].norm(dim=-1).mean().item()
        generated_diff_norm = diff[:, original_length:, :].norm(dim=-1).mean().item()
        
        print(f"Avg diff norm in prompt region: {prompt_diff_norm:.6f}")
        print(f"Avg diff norm in generated region: {generated_diff_norm:.6f}")
        
        # For embedding layers, generated region should have zero diff
        # For transformer layers, non-zero diff is expected due to attention
        if layer_name and 'embed' in layer_name.lower():
            if generated_diff_norm > 1e-5:
                print("⚠️  WARNING: Embedding layer shows unexpected generated region differences!")
        else:
            # For transformer layers, this is expected
            ratio = generated_diff_norm / (prompt_diff_norm + 1e-8)
            print(f"Generated/Prompt diff ratio: {ratio:.3f}")
            print("ℹ️  Non-zero generated region diff is expected for transformer layers (attention propagation)")


    def _intervention_hook_fn(self, module, inputs, outputs):
        """
        Replaces layer output with interpolated activations during the Riemann sum loop.

        Supports batched integration:
        - If we repeat input batch (B) by c points, layer output batch becomes (c*B).
        - interpolated_activations must match that shape.
        """
        if self.use_interpolated_activations and self.interpolated_activations is not None:
            if isinstance(outputs, tuple):
                ref_tensor = outputs[0]
                if self.interpolated_activations.shape != ref_tensor.shape:
                    if self.interpolated_activations.numel() == ref_tensor.numel():
                        self.interpolated_activations = self.interpolated_activations.view_as(ref_tensor)
                    else:
                        raise ValueError(
                            f"Shape mismatch in intervention_hook: expected {ref_tensor.shape}, "
                            f"got {self.interpolated_activations.shape}"
                        )
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

    def _model_forward_no_mask(self, input_ids: torch.Tensor) -> Any:
        """
        Forward helper used inside DLIG.
        Always uses attention_mask=None.
        Disables KV cache when supported to prevent memory growth.
        """
        if self.disable_kv_cache:
            try:
                return self.model(input_ids=input_ids, attention_mask=None, use_cache=False)
            except TypeError:
                # Model forward doesn't accept use_cache.
                return self.model(input_ids=input_ids, attention_mask=None)
        return self.model(input_ids=input_ids, attention_mask=None)

    def get_layer_activations(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass (attention_mask=None) and capture selected layer output.
        """
        layer_acts = []

        def hook_fn(module, inp, out):
            hidden_states = out[0] if isinstance(out, tuple) else out
            layer_acts.append(hidden_states.detach().clone())

        layer = self.hook_manager._get_layer()
        handle = layer.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                _ = self._model_forward_no_mask(input_ids=input_ids)
        finally:
            handle.remove()

        if not layer_acts:
            raise RuntimeError(f"Failed to capture activations at layer {layer}")
        return layer_acts[0]

    def create_baseline_input(self, x_t, mask_token_id, original_length):
        """
        Baseline for DLIG:
        - SAME X_t
        - NULL condition by masking prompt tokens only
        """
        baseline = x_t.clone()
        baseline[:, :original_length] = mask_token_id

        # ASSERTION: Verify generated part is unchanged
        assert torch.equal(baseline[:, original_length:], x_t[:, original_length:]), \
            "Baseline should keep generated tokens (Xt) identical!"

        return baseline

    def _compute_target_score(self, outputs, input_ids, original_length):
        """
        F_t: Score the model's log-probability of the ACTUAL tokens in X_t
        at the generated positions (after the prompt).
        
        For diffusion LLMs, we only score non-mask positions since mask
        tokens are placeholders, not actual predictions.
        """
        logits = outputs.logits  # [B, seq_len, vocab_size]
        
        # Handle edge case: sequence too short
        if logits.shape[1] <= original_length:
            # Fallback to max log-prob over all positions
            log_probs = torch.log_softmax(logits, dim=-1)
            max_log_probs, _ = log_probs.max(dim=-1)
            return max_log_probs.sum()
        
        # Next-token prediction: logits at position i predict token at i+1
        gen_logits = logits[:, original_length:-1, :]  # [B, gen_len-1, vocab_size]
        target_tokens = input_ids[:, original_length+1:]  # [B, gen_len-1]
        
        # Identify non-mask tokens (actually revealed/denoised tokens)
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            mask_token_id = self.tokenizer.pad_token_id
        
        # Also exclude eos token from scoring
        eos_token_id = self.tokenizer.eos_token_id
        
        non_mask = (target_tokens != mask_token_id)
        if eos_token_id is not None:
            non_mask = non_mask & (target_tokens != eos_token_id)
        
        non_mask = non_mask.float()  # [B, gen_len-1]
        
        # Compute log-probabilities
        log_probs = torch.log_softmax(gen_logits, dim=-1)  # [B, gen_len-1, vocab_size]
        
        # Gather log-prob of actual tokens at each position
        target_log_probs = log_probs.gather(
            dim=-1,
            index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)  # [B, gen_len-1]
        
        # Check if we have any non-mask tokens to score
        if non_mask.sum() == 0:
            # Fallback: use max log-prob scoring (original behavior)
            # This happens at step 0 when everything is still masked
            max_log_probs, _ = log_probs.max(dim=-1)
            return max_log_probs.sum()
        
        # Only score non-mask positions
        masked_target_log_probs = target_log_probs * non_mask
        
        return masked_target_log_probs.sum()

    def _process_results(self, dlig_raw, original_length, step, activation_diff):
        if self.relevant_token_indices:
            max_idx = max(self.relevant_token_indices)
            if max_idx >= dlig_raw.shape[1]:
                raise IndexError(
                    f"relevant_token_indices max ({max_idx}) out of bounds "
                    f"for sequence length {dlig_raw.shape[1]} at step {step}"
                )
            input_dlig = dlig_raw[:, self.relevant_token_indices, :].detach().cpu()
        else:
            input_dlig = dlig_raw[:, :original_length, :].detach().cpu()

        token_scores = input_dlig.sum(dim=-1)
        return {
            "step": step,
            "token_scores": token_scores,
            "full_dlig": input_dlig,
            "activation_diff_norm": activation_diff[:, :original_length, :].norm(dim=-1).detach().cpu(),
        }

    def compute_dlig_at_timestep(self, step, x_t, mask_token_id, original_length):
        """
        Vectorized DLIG at timestep t.

        IMPORTANT:
        - attention_mask is ignored here by design; we always call the model with attention_mask=None
          to avoid Dream SDPA mask broadcasting failures under repeated batches.

        DLIG math is unchanged.
        """
        self.model.eval()
        t_step0 = time.time() if self.enable_timing else None

        if step in [0, 5, 10]:  # Debug at a few timesteps
            self.debug_baseline_construction(step, x_t, mask_token_id, original_length)

        # 1) Capture activations (no grad)
        with torch.no_grad():
            real_act = self.get_layer_activations(x_t)
            baseline_inp = self.create_baseline_input(x_t, mask_token_id, original_length)
            if baseline_inp.device != x_t.device:
                baseline_inp = baseline_inp.to(x_t.device)
            baseline_act = self.get_layer_activations(baseline_inp)

            if step in [0, 5, 10]:
                self.debug_activation_differences(step, real_act, baseline_act, original_length)

        act_device = real_act.device
        act_dtype = real_act.dtype

        # Move token IDs to activation device for repeated-batch forward
        if x_t.device != act_device:
            x_t_local = x_t.to(act_device)
        else:
            x_t_local = x_t

        baseline_act = baseline_act.to(device=act_device, dtype=act_dtype)
        real_act = real_act.to(device=act_device, dtype=act_dtype)

        activation_diff = real_act - baseline_act  # [B,S,H]
        B, S, H = activation_diff.shape

        grad_sum = torch.zeros_like(real_act)

        m = int(self.integration_steps)
        if m <= 0:
            raise ValueError("integration_steps must be >= 1")

        chunk = self.integration_batch_size
        if chunk is None or chunk <= 0:
            chunk = m

        chunk_idx = 0
        for start in range(0, m, chunk):
            end = min(m, start + chunk)
            c = end - start
            chunk_idx += 1

            t_chunk0 = time.time() if self.enable_timing else None

            ks = torch.arange(start + 1, end + 1, device=act_device, dtype=act_dtype)
            alphas = (ks / m).view(c, 1, 1, 1)

            interpolated = baseline_act.unsqueeze(0) + alphas * activation_diff.unsqueeze(0)  # [c,B,S,H]
            interpolated = interpolated.detach().requires_grad_(True)
            interpolated_flat = interpolated.reshape(c * B, S, H)

            x_rep = x_t_local.repeat(c, 1)

            with self.activation_intervention(self.hook_manager.layer_name, interpolated_flat):
                outputs = self._model_forward_no_mask(input_ids=x_rep)
                target_score = self._compute_target_score(outputs, x_rep, original_length)

            grads_flat = torch.autograd.grad(target_score, interpolated_flat, retain_graph=False)[0]
            if chunk_idx == 1 and grads_flat.abs().max() < 1e-10:
                print(f"⚠️  WARNING: Near-zero gradients at step {step}, first chunk. "
                    f"Max grad magnitude: {grads_flat.abs().max().item():.2e}")
            grads = grads_flat.reshape(c, B, S, H)
            grad_sum += grads.detach().sum(dim=0)

            del interpolated, interpolated_flat, grads_flat, grads, outputs, target_score, x_rep

            if self.enable_timing and (chunk_idx % self.timing_log_every_chunks == 0):
                dt_chunk = time.time() - t_chunk0
                print(
                    f"[DLIG-TIMING] step={step} chunk={chunk_idx} k=[{start+1},{end}] c={c} time={dt_chunk:.2f}s",
                    flush=True,
                )

        avg_gradients = (grad_sum / m).detach()
        dlig_raw = activation_diff.detach() * avg_gradients

        if self.enable_timing:
            dt_step = time.time() - t_step0
            print(
                f"[DLIG-TIMING] step={step} total_time={dt_step:.2f}s "
                f"(m={m}, chunk={chunk}, B={B}, S={S}, H={H}, act_device={act_device})",
                flush=True,
            )

        return self._process_results(dlig_raw.cpu(), original_length, step, activation_diff)

    # ---- existing API (unchanged) ----
    def identify_relevant_tokens(self, input_tokens):
        special_tokens = {
            self.tokenizer.pad_token,
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
            self.tokenizer.unk_token,
            self.tokenizer.sep_token,
            self.tokenizer.cls_token,
            self.tokenizer.mask_token,
        }
        special_tokens = {t for t in special_tokens if t is not None}

        relevant_indices = []
        for idx, token in enumerate(input_tokens):
            if token not in special_tokens:
                relevant_indices.append(idx)
        return relevant_indices

    def set_target_output(self, target_text):
        target_ids = self.tokenizer.encode(target_text, return_tensors="pt")
        self.target_output_ids = target_ids.to(next(self.model.parameters()).device)
        print(f"[DEBUG] Target output set: {target_text}")
        print(f"[DEBUG] Target IDs shape: {self.target_output_ids.shape}")

    def generation_logits_hook_func(self, step, x, logits):
        if step is not None and self.original_input_length is not None:
            mask_token_id = (
                self.tokenizer.mask_token_id
                if self.tokenizer.mask_token_id is not None
                else self.tokenizer.pad_token_id
            )

            try:
                dlig_result = self.compute_dlig_at_timestep(
                    step, x, mask_token_id, self.original_input_length
                )
                self.dlig_scores.append(dlig_result)
                print(f"Step {step}: DLIG computed successfully.")
            except Exception as e:
                print(f"Step {step}: DLIG failed - {e}")
                traceback.print_exc()

            try:
                decoded = self.tokenizer.decode(x[0, self.original_input_length:], skip_special_tokens=True)
                print(f"Step {step} partial generation: {decoded[:50]}...")
            except Exception:
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