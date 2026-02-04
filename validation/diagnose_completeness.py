# validation/diagnose_completeness.py
"""
Diagnostic script to investigate the completeness axiom failure.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import MODEL_PATH
from models.model_manager import ModelManager
from attribution.dlig_attribution import DLIGAttribution
from attribution.hook_manager import HookManager


def diagnose_completeness(
    model,
    tokenizer,
    device,
    layer_idx: int = 7,
    integration_steps: int = 20,
):
    """Diagnose the completeness axiom failure."""
    
    print("=" * 70)
    print("COMPLETENESS AXIOM DIAGNOSTIC")
    print("=" * 70)
    
    # Create test input
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How to promote hate speech?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to(device)
    original_length = input_ids.shape[1]
    
    # Add generated response
    gen_tokens = tokenizer.encode(" I cannot help.", add_special_tokens=False)
    gen_tensor = torch.tensor([gen_tokens], device=device)
    x_t = torch.cat([input_ids, gen_tensor], dim=1)
    
    print(f"\nInput shape: {x_t.shape}")
    print(f"Prompt length: {original_length}")
    print(f"Generated length: {x_t.shape[1] - original_length}")
    
    # Setup
    target_layer = model.model.layers[layer_idx]
    hook_manager = HookManager(model)
    hook_manager.register_hook(target_layer)
    
    dlig = DLIGAttribution(
        model, tokenizer, hook_manager,
        integration_steps=integration_steps,
        integration_batch_size=5,
    )
    dlig.set_original_input_length(original_length)
    
    mask_id = tokenizer.mask_token_id or tokenizer.pad_token_id
    
    # ============================================================
    # DIAGNOSTIC 1: Check scoring function values
    # ============================================================
    print("\n" + "-" * 70)
    print("DIAGNOSTIC 1: Scoring Function Analysis")
    print("-" * 70)
    
    baseline = dlig.create_baseline_input(x_t, mask_id, original_length)
    
    with torch.no_grad():
        # Real prompt
        out_real = model(x_t)
        score_real = dlig._compute_target_score(out_real, x_t, original_length)
        
        # Baseline (masked prompt)
        out_base = model(baseline)
        score_base = dlig._compute_target_score(out_base, baseline, original_length)
    
    print(f"F(real prompt): {score_real.item():.6f}")
    print(f"F(masked baseline): {score_base.item():.6f}")
    print(f"F(real) - F(baseline): {(score_real - score_base).item():.6f}")
    
    # Check which is higher
    if score_real > score_base:
        print("→ Real prompt has HIGHER score (expected for good scoring)")
    else:
        print("→ Baseline has HIGHER score (unexpected!)")
        print("  This suggests the model is MORE confident without the prompt context.")
    
    # ============================================================
    # DIAGNOSTIC 2: Check gradient direction at endpoints
    # ============================================================
    print("\n" + "-" * 70)
    print("DIAGNOSTIC 2: Gradient Direction at Endpoints")
    print("-" * 70)
    
    # Get activations
    real_act = dlig.get_layer_activations(x_t)
    baseline_act = dlig.get_layer_activations(baseline)
    
    activation_diff = real_act - baseline_act
    print(f"Activation diff norm: {activation_diff.norm().item():.4f}")
    print(f"Activation diff mean: {activation_diff.mean().item():.6f}")
    
    # Compute gradient at real activation (alpha=1)
    real_act_grad = real_act.clone().detach().requires_grad_(True)
    
    def intervention_hook(module, inp, out):
        if isinstance(out, tuple):
            return (real_act_grad,) + out[1:]
        return real_act_grad
    
    handle = target_layer.register_forward_hook(intervention_hook)
    try:
        out = model(x_t)
        score = dlig._compute_target_score(out, x_t, original_length)
        grad_at_real = torch.autograd.grad(score, real_act_grad)[0]
    finally:
        handle.remove()
    
    print(f"\nGradient at real (alpha=1):")
    print(f"  Norm: {grad_at_real.norm().item():.6f}")
    print(f"  Mean: {grad_at_real.mean().item():.6f}")
    print(f"  Max: {grad_at_real.max().item():.6f}")
    print(f"  Min: {grad_at_real.min().item():.6f}")
    
    # Compute gradient at baseline activation (alpha=0)
    baseline_act_grad = baseline_act.clone().detach().requires_grad_(True)
    
    def intervention_hook_base(module, inp, out):
        if isinstance(out, tuple):
            return (baseline_act_grad,) + out[1:]
        return baseline_act_grad
    
    handle = target_layer.register_forward_hook(intervention_hook_base)
    try:
        out = model(x_t)
        score = dlig._compute_target_score(out, x_t, original_length)
        grad_at_base = torch.autograd.grad(score, baseline_act_grad)[0]
    finally:
        handle.remove()
    
    print(f"\nGradient at baseline (alpha=0):")
    print(f"  Norm: {grad_at_base.norm().item():.6f}")
    print(f"  Mean: {grad_at_base.mean().item():.6f}")
    print(f"  Max: {grad_at_base.max().item():.6f}")
    print(f"  Min: {grad_at_base.min().item():.6f}")
    
    # ============================================================
    # DIAGNOSTIC 3: Manual IG computation
    # ============================================================
    print("\n" + "-" * 70)
    print("DIAGNOSTIC 3: Manual Integrated Gradients")
    print("-" * 70)
    
    # Compute IG manually with explicit steps
    m = 10
    grad_sum = torch.zeros_like(real_act)
    
    for k in range(1, m + 1):
        alpha = k / m
        interpolated = baseline_act + alpha * activation_diff
        interpolated = interpolated.clone().detach().requires_grad_(True)
        
        def intervention_hook_interp(module, inp, out):
            if isinstance(out, tuple):
                return (interpolated,) + out[1:]
            return interpolated
        
        handle = target_layer.register_forward_hook(intervention_hook_interp)
        try:
            out = model(x_t)
            score = dlig._compute_target_score(out, x_t, original_length)
            grad = torch.autograd.grad(score, interpolated)[0]
            grad_sum += grad.detach()
        finally:
            handle.remove()
    
    avg_grad = grad_sum / m
    
    # Compute attribution
    ig_attribution = activation_diff * avg_grad
    ig_sum = ig_attribution.sum().item()
    
    print(f"Manual IG sum: {ig_sum:.6f}")
    print(f"Expected (F(x) - F(baseline)): {(score_real - score_base).item():.6f}")
    print(f"Ratio: {ig_sum / (score_real - score_base).item():.4f}")
    
    # ============================================================
    # DIAGNOSTIC 4: Check if gradients are computed correctly
    # ============================================================
    print("\n" + "-" * 70)
    print("DIAGNOSTIC 4: Gradient Sanity Check")
    print("-" * 70)
    
    # The dot product of gradient and activation_diff should tell us direction
    dot_product = (avg_grad * activation_diff).sum().item()
    print(f"avg_grad · activation_diff = {dot_product:.6f}")
    
    if dot_product > 0:
        print("→ Gradient and activation_diff point in SAME direction")
        print("  This means moving from baseline→real INCREASES the score")
    else:
        print("→ Gradient and activation_diff point in OPPOSITE directions")
        print("  This means moving from baseline→real DECREASES the score")
    
    # Compare with actual score change
    actual_change = (score_real - score_base).item()
    print(f"\nActual score change: {actual_change:.6f}")
    
    if (dot_product > 0 and actual_change > 0) or (dot_product < 0 and actual_change < 0):
        print("✓ Gradient direction MATCHES actual score change")
    else:
        print("✗ Gradient direction CONTRADICTS actual score change!")
        print("  This indicates the intervention hook may not be working correctly,")
        print("  or there's a path through the network that bypasses the hooked layer.")
    
    # ============================================================
    # DIAGNOSTIC 5: Check DLIG's actual computation
    # ============================================================
    print("\n" + "-" * 70)
    print("DIAGNOSTIC 5: DLIG Implementation Check")
    print("-" * 70)
    
    dlig_result = dlig.compute_dlig_at_timestep(
        step=0, x_t=x_t, mask_token_id=mask_id, original_length=original_length
    )
    
    dlig_sum = dlig_result["token_scores"].sum().item()
    full_dlig_sum = dlig_result["full_dlig"].sum().item()
    
    print(f"DLIG token_scores sum: {dlig_sum:.6f}")
    print(f"DLIG full_dlig sum: {full_dlig_sum:.6f}")
    print(f"Manual IG sum: {ig_sum:.6f}")
    print(f"Actual F(x) - F(baseline): {actual_change:.6f}")
    
    # Check if DLIG matches manual IG
    if abs(dlig_sum - ig_sum) < 0.1 * abs(ig_sum):
        print("✓ DLIG matches manual IG computation")
    else:
        print(f"✗ DLIG differs from manual IG by {abs(dlig_sum - ig_sum):.6f}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    issues = []
    
    if score_base > score_real:
        issues.append("Baseline scores HIGHER than real - scoring function may be inverted")
    
    if (dot_product > 0) != (actual_change > 0):
        issues.append("Gradient direction doesn't match actual score change - intervention may be broken")
    
    if abs(dlig_sum - ig_sum) > 0.1 * abs(max(ig_sum, 1e-6)):
        issues.append("DLIG implementation differs from manual IG")
    
    if not issues:
        print("No obvious issues found. The completeness violation may be due to:")
        print("  1. Numerical integration error (try more steps)")
        print("  2. The intervention hook not fully capturing the layer's effect")
        print("  3. Skip connections or other paths bypassing the hooked layer")
    else:
        print("Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    
    hook_manager.remove_hook()
    
    return {
        "score_real": score_real.item(),
        "score_base": score_base.item(),
        "actual_diff": actual_change,
        "manual_ig_sum": ig_sum,
        "dlig_sum": dlig_sum,
        "dot_product": dot_product,
    }


def main():
    print("Loading model...")
    mm = ModelManager(model_path=str(MODEL_PATH), device_map="auto", torch_dtype="float32")
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()
    print(f"Model loaded on device: {device}")
    
    results = diagnose_completeness(
        model, tokenizer, device,
        layer_idx=7,
        integration_steps=20,
    )
    
    print("\n" + "=" * 70)
    print("RAW RESULTS")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()