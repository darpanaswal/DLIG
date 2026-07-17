"""
Spot-check: partial-forward DLIG == full-path DLIG on the REAL model.
Run on one prompt, a few layers, one step. Compares full_dlig tensors.

This is the check the toy test could NOT do (toy has no attention / position mixing).
Run it ONCE before trusting the sped-up contrastive_attribution.py at scale.

Usage (single GPU):
    python -m experiments.theorems.spot_check_partial
or place next to verify_completeness.py and adapt the import path.
"""
import torch
from attribution.hook_manager import MultiLayerHookManager
from attribution.dlig_attribution import DLIGAttribution
from utils.config import MODEL_PATH
from models.model_manager import ModelManager, GradientEnabledModel
# reuse trajectory recorder from the verification harness
from experiments.theorems.verify_completeness import TrajRecorder, set_seed

LAYERS   = ["0", "8", "16", "25"]   # span shallow -> deep
STEP     = 2                         # mid-trajectory (well-behaved F)
M        = 12                        # YOUR production m
CHUNK    = 12
GEN_STEPS = 8
PROMPT   = "Explain how photosynthesis works."

def main():
    set_seed(0)
    mm = ModelManager(model_path=str(MODEL_PATH), device_map="auto",
                      torch_dtype=torch.bfloat16)   # MATCH production dtype
    model, tok = mm.load_model_and_tokenizer()
    device = mm.get_model_device()

    mask_id = int(tok.mask_token_id or tok.pad_token_id or 0)

    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT}]
    inputs = tok.apply_chat_template(messages, return_tensors="pt",
                                     return_dict=True, add_generation_prompt=True)
    input_ids = inputs.input_ids.to(device)
    attn = inputs.attention_mask.to(device).float()
    L = int(input_ids.shape[1])

    gm = GradientEnabledModel(model)
    rec = TrajRecorder()
    _ = gm.diffusion_generate_with_grad(
        input_ids, attention_mask=attn,
        max_new_tokens=64, steps=GEN_STEPS,
        generation_logits_hook_func=rec.hook,
    )
    x_t = rec.x_by_step[STEP].to(device)

    mlhm = MultiLayerHookManager(model, layer_specs=LAYERS)
    baseline_inp = x_t.clone(); baseline_inp[:, :L] = mask_id
    with torch.no_grad():
        real = mlhm.capture_activations(x_t, disable_kv_cache=True)
        base = mlhm.capture_activations(baseline_inp, disable_kv_cache=True)

    print(f"{'layer':>6} {'max_abs_diff':>14} {'rel_diff':>12}  verdict")
    worst = 0.0
    for spec in LAYERS:
        outs = {}
        for use_partial in (False, True):
            dlig = DLIGAttribution(
                model, tok, mlhm.get_layer_view(spec),
                integration_steps=M, integration_batch_size=CHUNK,
                disable_kv_cache=True, score_mode="logprob",
                use_partial_forward=use_partial,
            )
            dlig.set_original_input_length(L)
            dlig.target_output_ids = None
            res = dlig.compute_dlig_at_timestep_with_activations(
                step=STEP, x_t=x_t,
                real_act=real[spec], baseline_act=base[spec], original_length=L,
            )
            outs[use_partial] = res["full_dlig"][0].sum(dim=-1).float()
        d = (outs[True] - outs[False]).abs().max().item()
        rel = d / (outs[False].abs().max().item() + 1e-12)
        worst = max(worst, rel)
        tag = "OK" if rel < 1e-2 else "*** INVESTIGATE ***"
        print(f"{spec:>6} {d:>14.3e} {rel:>12.3e}  {tag}")
    print(f"\nworst rel_diff = {worst:.3e}  "
          f"({'PASS - bf16 roundoff only' if worst < 1e-2 else 'FAIL - real discrepancy'})")

if __name__ == "__main__":
    main()