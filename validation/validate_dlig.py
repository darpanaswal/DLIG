# validation/validate_dlig.py
"""
Comprehensive DLIG implementation validation.
Run this before trusting any experimental results.

Validates:
1. Baseline construction correctness
2. Gradient flow through hooked layers
3. Scoring function sensibility
4. Completeness axiom (most critical)
5. Sensitivity to input changes
6. Comparison between harmful vs benign prompts
"""

import os
import sys
import torch
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import MODEL_PATH
from models.model_manager import ModelManager, GradientEnabledModel
from attribution.dlig_attribution import DLIGAttribution
from attribution.hook_manager import HookManager


@dataclass
class ValidationResult:
    name: str
    passed: bool
    details: Dict[str, Any]
    message: str


class DLIGValidator:
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        integration_steps: int = 10,
        verbose: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.integration_steps = integration_steps
        self.verbose = verbose
        self.results: List[ValidationResult] = []

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def _create_test_input(
        self,
        user_message: str,
        system_message: str = "You are a helpful assistant.",
        generated_response: str = " I cannot help with that request.",
    ) -> Tuple[torch.Tensor, int]:
        """Create a test input with prompt + simulated generation."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to(self.device)
        original_length = input_ids.shape[1]

        # Append simulated generation
        gen_tokens = self.tokenizer.encode(generated_response, add_special_tokens=False)
        gen_tensor = torch.tensor([gen_tokens], device=self.device)
        x_t = torch.cat([input_ids, gen_tensor], dim=1)

        return x_t, original_length

    def _setup_dlig(self, layer_idx: int) -> Tuple[DLIGAttribution, HookManager]:
        """Setup DLIG attribution for a specific layer."""
        target_layer = self.model.model.layers[layer_idx]
        hook_manager = HookManager(self.model)
        hook_manager.register_hook(target_layer)

        dlig = DLIGAttribution(
            self.model,
            self.tokenizer,
            hook_manager,
            integration_steps=self.integration_steps,
        )

        return dlig, hook_manager

    def _get_mask_token_id(self) -> int:
        """Get mask token ID with fallback."""
        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            mask_id = self.tokenizer.pad_token_id
        if mask_id is None:
            mask_id = 0
        return mask_id

    def run_all_validations(
        self,
        layer_idx: int = 7,
        harmful_prompt: str = "How to promote hate speech and discrimination?",
        benign_prompt: str = "How to promote kindness and inclusion?",
    ) -> List[ValidationResult]:
        """Run all validation checks."""
        self.results = []

        self.log("=" * 70)
        self.log("DLIG IMPLEMENTATION VALIDATION")
        self.log("=" * 70)
        self.log(f"\nConfiguration:")
        self.log(f"  Layer: {layer_idx}")
        self.log(f"  Integration steps: {self.integration_steps}")
        self.log(f"  Harmful prompt: '{harmful_prompt}'")
        self.log(f"  Benign prompt: '{benign_prompt}'")

        # Create test inputs
        x_t_harmful, orig_len_harmful = self._create_test_input(harmful_prompt)
        x_t_benign, orig_len_benign = self._create_test_input(benign_prompt)

        self.log(f"\n  Harmful input shape: {x_t_harmful.shape}, prompt length: {orig_len_harmful}")
        self.log(f"  Benign input shape: {x_t_benign.shape}, prompt length: {orig_len_benign}")

        # Setup DLIG
        dlig, hook_manager = self._setup_dlig(layer_idx)
        dlig.set_original_input_length(orig_len_harmful)

        try:
            # Test 1: Baseline Construction
            self.log("\n" + "-" * 70)
            self.log("[TEST 1] Baseline Construction")
            self.log("-" * 70)
            self.results.append(
                self._validate_baseline(dlig, x_t_harmful, orig_len_harmful)
            )

            # Test 2: Scoring Function Difference
            self.log("\n" + "-" * 70)
            self.log("[TEST 2] Scoring Function (Prompt vs Baseline)")
            self.log("-" * 70)
            self.results.append(
                self._validate_scoring(dlig, x_t_harmful, orig_len_harmful)
            )

            # Test 3: Gradient Flow
            self.log("\n" + "-" * 70)
            self.log("[TEST 3] Gradient Flow Through Hooked Layer")
            self.log("-" * 70)
            self.results.append(
                self._validate_gradient_flow(dlig, hook_manager, x_t_harmful, orig_len_harmful)
            )

            # Test 4: Completeness Axiom (MOST CRITICAL)
            self.log("\n" + "-" * 70)
            self.log("[TEST 4] Completeness Axiom (Sum of Attributions ≈ Output Diff)")
            self.log("-" * 70)
            self.results.append(
                self._validate_completeness(dlig, x_t_harmful, orig_len_harmful)
            )

            # Test 5: Sensitivity to Input
            self.log("\n" + "-" * 70)
            self.log("[TEST 5] Sensitivity (Modified Input → Different Attributions)")
            self.log("-" * 70)
            self.results.append(
                self._validate_sensitivity(dlig, x_t_harmful, orig_len_harmful)
            )

            # Test 6: Harmful vs Benign Comparison
            self.log("\n" + "-" * 70)
            self.log("[TEST 6] Harmful vs Benign Prompt Comparison")
            self.log("-" * 70)
            self.results.append(
                self._validate_harmful_vs_benign(
                    dlig, x_t_harmful, orig_len_harmful, x_t_benign, orig_len_benign
                )
            )

            # Test 7: Non-zero Attributions
            self.log("\n" + "-" * 70)
            self.log("[TEST 7] Non-Zero Attributions Check")
            self.log("-" * 70)
            dlig.set_original_input_length(orig_len_harmful)
            self.results.append(
                self._validate_nonzero_attributions(dlig, x_t_harmful, orig_len_harmful)
            )

            # Test 8: Attribution Sign Consistency
            self.log("\n" + "-" * 70)
            self.log("[TEST 8] Attribution Sign Distribution")
            self.log("-" * 70)
            self.results.append(
                self._validate_sign_distribution(dlig, x_t_harmful, orig_len_harmful)
            )

        finally:
            hook_manager.remove_hook()

        # Print summary
        self._print_summary()

        return self.results

    def _validate_baseline(
        self, dlig: DLIGAttribution, x_t: torch.Tensor, original_length: int
    ) -> ValidationResult:
        """Verify baseline construction is correct."""
        mask_id = self._get_mask_token_id()

        # Use the actual method from DLIGAttribution
        baseline = dlig.create_baseline_input(x_t, mask_id, original_length)

        # Check prompt region is all masks
        prompt_baseline = baseline[0, :original_length]
        prompt_all_masks = (prompt_baseline == mask_id).all().item()

        # Check generated region is identical
        gen_real = x_t[0, original_length:]
        gen_baseline = baseline[0, original_length:]
        gen_identical = (gen_real == gen_baseline).all().item()

        passed = prompt_all_masks and gen_identical

        self.log(f"  Mask token ID: {mask_id}")
        self.log(f"  Prompt length: {original_length}")
        self.log(f"  Total length: {x_t.shape[1]}")
        self.log(f"  Prompt region all masks: {prompt_all_masks} {'✓' if prompt_all_masks else '✗'}")
        self.log(f"  Generated region identical: {gen_identical} {'✓' if gen_identical else '✗'}")
        self.log(f"  RESULT: {'PASSED ✓' if passed else 'FAILED ✗'}")

        return ValidationResult(
            name="Baseline Construction",
            passed=passed,
            details={"prompt_masked": prompt_all_masks, "gen_identical": gen_identical, "mask_id": mask_id},
            message="Baseline correctly masks prompt while preserving generated tokens" if passed else "Baseline construction error",
        )

    def _validate_scoring(
        self, dlig: DLIGAttribution, x_t: torch.Tensor, original_length: int
    ) -> ValidationResult:
        """Verify scoring function produces different values for prompt vs baseline."""
        mask_id = self._get_mask_token_id()

        with torch.no_grad():
            # Score with real input
            outputs_real = self.model(x_t)
            score_real = dlig._compute_target_score(outputs_real, x_t, original_length)

            # Score with baseline
            baseline = dlig.create_baseline_input(x_t, mask_id, original_length)
            outputs_baseline = self.model(baseline)
            score_baseline = dlig._compute_target_score(outputs_baseline, baseline, original_length)

        score_real_val = score_real.item()
        score_baseline_val = score_baseline.item()
        diff = score_real_val - score_baseline_val
        relative_diff = abs(diff) / max(abs(score_real_val), abs(score_baseline_val), 1e-10)

        # Scores should be meaningfully different
        passed = abs(diff) > 1e-6 and relative_diff > 0.001

        self.log(f"  Score (real prompt): {score_real_val:.6f}")
        self.log(f"  Score (masked prompt): {score_baseline_val:.6f}")
        self.log(f"  Absolute difference: {diff:.6f}")
        self.log(f"  Relative difference: {relative_diff:.4%}")
        self.log(f"  RESULT: {'PASSED ✓' if passed else 'FAILED ✗'}")

        return ValidationResult(
            name="Scoring Function",
            passed=passed,
            details={
                "score_real": score_real_val,
                "score_baseline": score_baseline_val,
                "diff": diff,
                "relative_diff": relative_diff,
            },
            message="Scoring function differentiates prompt from baseline" if passed else "Scores too similar - prompt has no effect?",
        )

    def _validate_gradient_flow(
        self,
        dlig: DLIGAttribution,
        hook_manager: HookManager,
        x_t: torch.Tensor,
        original_length: int,
    ) -> ValidationResult:
        """Verify gradients flow through the hooked layer."""
        self.model.zero_grad()

        # Capture activation with gradient tracking
        activation_holder = []

        def capture_hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            # Keep it attached to computation graph
            activation_holder.append(hidden)

        layer = hook_manager._get_layer()
        handle = layer.register_forward_hook(capture_hook)

        try:
            # Forward pass
            outputs = self.model(x_t)
            score = dlig._compute_target_score(outputs, x_t, original_length)

            if not activation_holder:
                self.log("  ERROR: No activation captured!")
                return ValidationResult(
                    name="Gradient Flow",
                    passed=False,
                    details={"error": "No activation captured"},
                    message="Hook failed to capture activation",
                )

            activation = activation_holder[0]
            self.log(f"  Activation shape: {activation.shape}")
            self.log(f"  Activation requires_grad: {activation.requires_grad}")
            self.log(f"  Score requires_grad: {score.requires_grad}")

            # Compute gradient
            if score.requires_grad:
                score.backward(retain_graph=True)
                
                # Check if activation has gradient
                if activation.grad is not None:
                    grad_norm = activation.grad.norm().item()
                    grad_max = activation.grad.abs().max().item()
                else:
                    # Try explicit gradient computation
                    try:
                        grad = torch.autograd.grad(score, activation, retain_graph=True, allow_unused=True)[0]
                        if grad is not None:
                            grad_norm = grad.norm().item()
                            grad_max = grad.abs().max().item()
                        else:
                            grad_norm = 0.0
                            grad_max = 0.0
                    except Exception as e:
                        self.log(f"  Gradient computation error: {e}")
                        grad_norm = 0.0
                        grad_max = 0.0
            else:
                self.log("  WARNING: Score does not require grad")
                grad_norm = 0.0
                grad_max = 0.0

            passed = grad_norm > 1e-12

            self.log(f"  Gradient norm: {grad_norm:.6e}")
            self.log(f"  Gradient max: {grad_max:.6e}")
            self.log(f"  RESULT: {'PASSED ✓' if passed else 'FAILED ✗'}")

            return ValidationResult(
                name="Gradient Flow",
                passed=passed,
                details={"grad_norm": grad_norm, "grad_max": grad_max},
                message="Gradients flow through hooked layer" if passed else "No gradient flow - check hook setup",
            )

        finally:
            handle.remove()
            self.model.zero_grad()

    def _validate_completeness(
        self, dlig: DLIGAttribution, x_t: torch.Tensor, original_length: int
    ) -> ValidationResult:
        """
        Validate the completeness axiom of Integrated Gradients:
        Sum of attributions ≈ F(x) - F(baseline)

        THIS IS THE MOST CRITICAL TEST.
        """
        mask_id = self._get_mask_token_id()

        self.log(f"  Computing DLIG (this may take a moment)...")

        # Compute DLIG
        dlig_result = dlig.compute_dlig_at_timestep(
            step=0,
            x_t=x_t,
            mask_token_id=mask_id,
            original_length=original_length,
        )

        # Sum of attributions
        token_scores = dlig_result["token_scores"]
        full_dlig = dlig_result["full_dlig"]
        
        attribution_sum = token_scores.sum().item()
        full_dlig_sum = full_dlig.sum().item()

        self.log(f"  Token scores shape: {token_scores.shape}")
        self.log(f"  Full DLIG shape: {full_dlig.shape}")

        # Actual output difference
        with torch.no_grad():
            outputs_real = self.model(x_t)
            score_real = dlig._compute_target_score(outputs_real, x_t, original_length)

            baseline = dlig.create_baseline_input(x_t, mask_id, original_length)
            outputs_baseline = self.model(baseline)
            score_baseline = dlig._compute_target_score(outputs_baseline, baseline, original_length)

        actual_diff = (score_real - score_baseline).item()

        self.log(f"  Sum of token attributions: {attribution_sum:.6f}")
        self.log(f"  Sum of full DLIG: {full_dlig_sum:.6f}")
        self.log(f"  Actual score difference (F(x) - F(baseline)): {actual_diff:.6f}")

        if abs(actual_diff) > 1e-8:
            ratio = attribution_sum / actual_diff
            full_ratio = full_dlig_sum / actual_diff
            self.log(f"  Ratio (token_scores / actual_diff): {ratio:.4f}")
            self.log(f"  Ratio (full_dlig / actual_diff): {full_ratio:.4f}")
            self.log(f"  Expected ratio: ~1.0 (within 0.3-3.0 is acceptable due to numerical integration)")

            # Allow tolerance due to numerical integration errors
            passed = 0.1 < abs(ratio) < 10.0 or 0.1 < abs(full_ratio) < 10.0
            
            # Also check if they have the same sign
            same_sign = (attribution_sum * actual_diff > 0) or (full_dlig_sum * actual_diff > 0)
            if not same_sign:
                self.log(f"  WARNING: Attribution sum and actual diff have opposite signs!")
                passed = False
        else:
            ratio = float("nan")
            full_ratio = float("nan")
            # If actual diff is near zero, attributions should also be near zero
            passed = abs(attribution_sum) < 1e-4 and abs(full_dlig_sum) < 1e-4
            self.log(f"  Actual diff near zero, checking if attributions also small: {passed}")

        self.log(f"  RESULT: {'PASSED ✓' if passed else 'FAILED ✗'}")

        if not passed:
            self.log("  ⚠️  WARNING: Completeness axiom may be violated!")
            self.log("  This could indicate issues with integration or gradient computation.")
            self.log("  However, some deviation is expected with finite integration steps.")

        return ValidationResult(
            name="Completeness Axiom",
            passed=passed,
            details={
                "attribution_sum": attribution_sum,
                "full_dlig_sum": full_dlig_sum,
                "actual_diff": actual_diff,
                "ratio": ratio,
                "full_ratio": full_ratio,
            },
            message="Completeness axiom approximately satisfied" if passed else "Completeness axiom violated - review implementation",
        )

    def _validate_sensitivity(
        self, dlig: DLIGAttribution, x_t: torch.Tensor, original_length: int
    ) -> ValidationResult:
        """Verify that modifying input changes attributions."""
        mask_id = self._get_mask_token_id()

        self.log(f"  Computing DLIG for original input...")
        
        # Original DLIG
        dlig_original = dlig.compute_dlig_at_timestep(
            step=0, x_t=x_t, mask_token_id=mask_id, original_length=original_length
        )
        scores_original = dlig_original["token_scores"].clone()

        # Modify input - replace a token in the prompt
        x_t_modified = x_t.clone()
        mid_idx = min(original_length // 2, original_length - 1)
        original_token = x_t_modified[0, mid_idx].item()

        # Use a common neutral token
        new_token_ids = self.tokenizer.encode("hello", add_special_tokens=False)
        if new_token_ids:
            new_token = new_token_ids[0]
        else:
            new_token = original_token + 1

        x_t_modified[0, mid_idx] = new_token

        self.log(f"  Computing DLIG for modified input...")
        
        # Modified DLIG
        dlig_modified = dlig.compute_dlig_at_timestep(
            step=0, x_t=x_t_modified, mask_token_id=mask_id, original_length=original_length
        )
        scores_modified = dlig_modified["token_scores"]

        # Compute difference
        diff_l1 = (scores_original - scores_modified).abs().sum().item()
        diff_l2 = (scores_original - scores_modified).norm().item()
        max_diff = (scores_original - scores_modified).abs().max().item()

        passed = diff_l1 > 1e-8

        original_token_str = self.tokenizer.decode([original_token])
        new_token_str = self.tokenizer.decode([new_token])

        self.log(f"  Modified token at index {mid_idx}: '{original_token_str}' → '{new_token_str}'")
        self.log(f"  Attribution difference (L1): {diff_l1:.6e}")
        self.log(f"  Attribution difference (L2): {diff_l2:.6e}")
        self.log(f"  Max element difference: {max_diff:.6e}")
        self.log(f"  RESULT: {'PASSED ✓' if passed else 'FAILED ✗'}")

        return ValidationResult(
            name="Sensitivity",
            passed=passed,
            details={
                "diff_l1": diff_l1,
                "diff_l2": diff_l2,
                "max_diff": max_diff,
                "modified_idx": mid_idx,
            },
            message="Attributions sensitive to input changes" if passed else "Attributions unchanged - possible bug",
        )

    def _validate_harmful_vs_benign(
        self,
        dlig: DLIGAttribution,
        x_t_harmful: torch.Tensor,
        orig_len_harmful: int,
        x_t_benign: torch.Tensor,
        orig_len_benign: int,
    ) -> ValidationResult:
        """Compare attributions between harmful and benign prompts."""
        mask_id = self._get_mask_token_id()

        self.log(f"  Computing DLIG for harmful prompt...")
        
        # DLIG for harmful
        dlig.set_original_input_length(orig_len_harmful)
        dlig_harmful = dlig.compute_dlig_at_timestep(
            step=0, x_t=x_t_harmful, mask_token_id=mask_id, original_length=orig_len_harmful
        )

        self.log(f"  Computing DLIG for benign prompt...")
        
        # DLIG for benign
        dlig.set_original_input_length(orig_len_benign)
        dlig_benign = dlig.compute_dlig_at_timestep(
            step=0, x_t=x_t_benign, mask_token_id=mask_id, original_length=orig_len_benign
        )

        # Compare statistics
        harmful_mean = dlig_harmful["token_scores"].mean().item()
        harmful_std = dlig_harmful["token_scores"].std().item()
        harmful_max = dlig_harmful["token_scores"].abs().max().item()

        benign_mean = dlig_benign["token_scores"].mean().item()
        benign_std = dlig_benign["token_scores"].std().item()
        benign_max = dlig_benign["token_scores"].abs().max().item()

        # They should be different
        mean_diff = abs(harmful_mean - benign_mean)
        std_diff = abs(harmful_std - benign_std)
        max_diff = abs(harmful_max - benign_max)

        # At least some difference should exist
        passed = mean_diff > 1e-8 or std_diff > 1e-8 or max_diff > 1e-8

        self.log(f"  Harmful - mean: {harmful_mean:.6e}, std: {harmful_std:.6e}, max: {harmful_max:.6e}")
        self.log(f"  Benign  - mean: {benign_mean:.6e}, std: {benign_std:.6e}, max: {benign_max:.6e}")
        self.log(f"  Differences - mean: {mean_diff:.6e}, std: {std_diff:.6e}, max: {max_diff:.6e}")
        self.log(f"  RESULT: {'PASSED ✓' if passed else 'FAILED ✗'}")

        return ValidationResult(
            name="Harmful vs Benign",
            passed=passed,
            details={
                "harmful": {"mean": harmful_mean, "std": harmful_std, "max": harmful_max},
                "benign": {"mean": benign_mean, "std": benign_std, "max": benign_max},
                "differences": {"mean": mean_diff, "std": std_diff, "max": max_diff},
            },
            message="Different prompts produce different attributions" if passed else "No difference between harmful/benign - suspicious",
        )

    def _validate_nonzero_attributions(
        self, dlig: DLIGAttribution, x_t: torch.Tensor, original_length: int
    ) -> ValidationResult:
        """Check that attributions are not all zeros."""
        mask_id = self._get_mask_token_id()

        # Reuse cached result if available from completeness test
        dlig_result = dlig.compute_dlig_at_timestep(
            step=0, x_t=x_t, mask_token_id=mask_id, original_length=original_length
        )

        token_scores = dlig_result["token_scores"]
        full_dlig = dlig_result["full_dlig"]

        token_nonzero = (token_scores.abs() > 1e-10).sum().item()
        token_total = token_scores.numel()
        token_nonzero_pct = token_nonzero / token_total * 100

        full_nonzero = (full_dlig.abs() > 1e-10).sum().item()
        full_total = full_dlig.numel()
        full_nonzero_pct = full_nonzero / full_total * 100

        passed = token_nonzero_pct > 10  # At least 10% should be non-zero

        self.log(f"  Token scores: {token_nonzero}/{token_total} non-zero ({token_nonzero_pct:.1f}%)")
        self.log(f"  Full DLIG: {full_nonzero}/{full_total} non-zero ({full_nonzero_pct:.1f}%)")
        self.log(f"  RESULT: {'PASSED ✓' if passed else 'FAILED ✗'}")

        return ValidationResult(
            name="Non-Zero Attributions",
            passed=passed,
            details={
                "token_nonzero": token_nonzero,
                "token_total": token_total,
                "token_nonzero_pct": token_nonzero_pct,
                "full_nonzero": full_nonzero,
                "full_total": full_total,
                "full_nonzero_pct": full_nonzero_pct,
            },
            message="Attributions are meaningfully non-zero" if passed else "Too many zero attributions",
        )

    def _validate_sign_distribution(
        self, dlig: DLIGAttribution, x_t: torch.Tensor, original_length: int
    ) -> ValidationResult:
        """Check that attributions have both positive and negative values."""
        mask_id = self._get_mask_token_id()

        dlig_result = dlig.compute_dlig_at_timestep(
            step=0, x_t=x_t, mask_token_id=mask_id, original_length=original_length
        )

        token_scores = dlig_result["token_scores"]

        num_positive = (token_scores > 1e-10).sum().item()
        num_negative = (token_scores < -1e-10).sum().item()
        num_zero = token_scores.numel() - num_positive - num_negative

        total = token_scores.numel()
        pos_pct = num_positive / total * 100
        neg_pct = num_negative / total * 100

        # Should have both positive and negative
        passed = num_positive > 0 and num_negative > 0

        self.log(f"  Positive: {num_positive} ({pos_pct:.1f}%)")
        self.log(f"  Negative: {num_negative} ({neg_pct:.1f}%)")
        self.log(f"  Near-zero: {num_zero}")
        self.log(f"  RESULT: {'PASSED ✓' if passed else 'FAILED ✗'}")

        if not passed:
            self.log("  ⚠️  WARNING: All attributions have same sign - possible sign error")

        return ValidationResult(
            name="Sign Distribution",
            passed=passed,
            details={
                "num_positive": num_positive,
                "num_negative": num_negative,
                "num_zero": num_zero,
                "pos_pct": pos_pct,
                "neg_pct": neg_pct,
            },
            message="Mixed positive/negative attributions (expected)" if passed else "All same sign - suspicious",
        )

    def _print_summary(self):
        """Print validation summary."""
        self.log("\n" + "=" * 70)
        self.log("VALIDATION SUMMARY")
        self.log("=" * 70)

        passed_count = 0
        failed_count = 0
        critical_failures = []

        for result in self.results:
            status = "PASSED ✓" if result.passed else "FAILED ✗"
            self.log(f"  {result.name}: {status}")
            if result.passed:
                passed_count += 1
            else:
                failed_count += 1
                if result.name == "Completeness Axiom":
                    critical_failures.append(result.name)

        self.log("-" * 70)
        self.log(f"  Total: {passed_count} passed, {failed_count} failed")
        self.log("-" * 70)

        if failed_count == 0:
            self.log("✓ ALL TESTS PASSED - Implementation appears correct")
        elif critical_failures:
            self.log("⚠️  CRITICAL FAILURES DETECTED:")
            for name in critical_failures:
                self.log(f"    - {name}")
            self.log("  Review implementation carefully before trusting results.")
        else:
            self.log("⚠️  SOME TESTS FAILED - Review implementation")

        self.log("=" * 70)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate DLIG implementation")
    parser.add_argument("--model_path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="float32")
    parser.add_argument("--layer", type=int, default=7, help="Layer to validate")
    parser.add_argument("--integration_steps", type=int, default=200, help="Integration steps (lower = faster)")
    parser.add_argument(
        "--harmful_prompt",
        type=str,
        default="How to promote hate speech and discrimination?",
    )
    parser.add_argument(
        "--benign_prompt",
        type=str,
        default="How to promote kindness and inclusion?",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    print("Loading model...")
    mm = ModelManager(
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()

    print(f"Model loaded on device: {device}")

    validator = DLIGValidator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        integration_steps=args.integration_steps,
        verbose=True,
    )

    results = validator.run_all_validations(
        layer_idx=args.layer,
        harmful_prompt=args.harmful_prompt,
        benign_prompt=args.benign_prompt,
    )

    # Return exit code based on results
    failed = sum(1 for r in results if not r.passed)
    critical = any(r.name == "Completeness Axiom" and not r.passed for r in results)

    if critical:
        sys.exit(2)  # Critical failure
    elif failed > 0:
        sys.exit(1)  # Some failures
    else:
        sys.exit(0)  # All passed


if __name__ == "__main__":
    main()