"""
Spot-check: partial-forward DLIG == full-path DLIG on the REAL model.
Run on one prompt, a few layers, one step. Compares full_dlig tensors.

This is the check the toy test could NOT do (toy has no attention / position mixing).
Run it ONCE before trusting the sped-up contrastive_attribution.py at scale.

Family-agnostic: uses the same ModelManager / backend abstraction as
verify_completeness.py, so it runs against Dream or DiffuGPT (base or any
fine-tuned checkpoint, via --model_path).

Usage:
    python -m experiments.theorems.spot_check_partial
    python -m experiments.theorems.spot_check_partial --family diffugpt \
        --model_path models/diffugpt-m-prosqa --layers 0 6 11
"""
import argparse
import torch

from models.backends import build_backend
from models.model_manager import ModelManager
from attribution.hook_manager import MultiLayerHookManager
from attribution.dlig_attribution import DLIGAttribution
# reuse trajectory recorder / prompt builder / seeding from the verification harness
from experiments.theorems.verify_completeness import (
    TrajRecorder, set_seed, build_prompt_inputs,
)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Partial-forward vs full-path DLIG spot check")
    p.add_argument("--family", type=str, default="dream", choices=["dream", "diffugpt"])
    p.add_argument("--model_path", type=str, default=None,
                   help="Override checkpoint dir (e.g. models/diffugpt-m-prosqa). "
                        "Defaults to DREAM_PATH/GPT_PATH based on --family.")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--torch_dtype", type=str, default="bfloat16",
                   help="MATCH production dtype.")
    p.add_argument("--layers", type=str, nargs="+", default=["0", "8", "16", "25"],
                   help="Layer indices (or 'embed_tokens') spanning shallow -> deep.")
    p.add_argument("--step", type=int, default=2, help="Mid-trajectory step (well-behaved F).")
    p.add_argument("--m", type=int, default=12, help="Integration steps (YOUR production m).")
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--gen_steps", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--prompt", type=str, default="Explain how photosynthesis works.")
    p.add_argument("--system", type=str, default="You are a helpful assistant.")
    p.add_argument("--score_mode", type=str, default="logprob",
                   choices=["meancentered", "logprob"])
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    mm = ModelManager(
        family=args.family,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        model_path=args.model_path,
    )
    model, tok = mm.load_model_and_tokenizer()
    device = mm.get_model_device()

    backend = build_backend(model, tok, family=args.family)
    print(f"[INFO] Backend family: {backend.family}  predicts_shifted={backend.predicts_shifted}")

    mask_id = tok.mask_token_id
    if mask_id is None:
        mask_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    mask_id = int(mask_id)

    input_ids, attn, L = build_prompt_inputs(tok, args.system, args.prompt, device)
    print(f"[INFO] prompt length L={L}")

    rec = TrajRecorder()
    _ = backend.generate_trajectory(
        input_ids,
        attention_mask=attn,
        max_new_tokens=args.max_new_tokens,
        steps=args.gen_steps,
        record_hook=rec.hook,
    )
    if args.step not in rec.x_by_step:
        raise ValueError(f"step {args.step} not among recorded steps {rec.steps()}")
    x_t = rec.x_by_step[args.step].to(device)

    mlhm = MultiLayerHookManager(model, layer_specs=args.layers, backend=backend)
    baseline_inp = x_t.clone()
    baseline_inp[:, :L] = mask_id
    with torch.no_grad():
        real = mlhm.capture_activations(x_t, disable_kv_cache=True)
        base = mlhm.capture_activations(baseline_inp, disable_kv_cache=True)

    print(f"{'layer':>6} {'max_abs_diff':>14} {'rel_diff':>12}  verdict")
    worst = 0.0
    for spec in args.layers:
        outs = {}
        for use_partial in (False, True):
            dlig = DLIGAttribution(
                model, tok, mlhm.get_layer_view(spec),
                integration_steps=args.m, integration_batch_size=args.chunk,
                disable_kv_cache=True, score_mode=args.score_mode,
                use_partial_forward=use_partial,
                backend=backend,
            )
            dlig.set_original_input_length(L)
            dlig.target_output_ids = None
            res = dlig.compute_dlig_at_timestep_with_activations(
                step=args.step, x_t=x_t,
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
