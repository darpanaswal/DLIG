#!/usr/bin/env python3
"""
verify_completeness_grid.py — completeness check over the FULL (layer, step) grid
in ONE process, instead of one `verify_completeness.py` subprocess per layer.

WHY THIS EXISTS
---------------
verify_completeness.py --layer L takes a single layer per invocation, so running
the full grid means N_LAYERS separate subprocesses, each of which:
  (a) reloads the model,
  (b) regenerates the trajectory from the SAME seed (identical result, wasted
      compute), and
  (c) extracts real_act/baseline_act via a single-layer forward
      (dlig.get_layer_activations), i.e. one full forward pass PER LAYER PER STEP
      just to read off one layer's activation.

(a) and (c) don't depend on which layer you intervene at -- MultiLayerHookManager
(already used by spot_check_partial.py) captures every analyzed layer's
activation in ONE forward pass. This script loads the model once, generates the
trajectory once, and per step does exactly two forward passes (real + baseline)
to get every layer's activation, then loops over layers only for the part that
truly cannot be shared: the m-step Riemann integration, which requires a hook
that INTERVENES at that specific layer (activation_intervention), so those
forward+backward passes are genuinely layer-specific and are NOT reduced here.

This does NOT use the partial-forward optimization (use_partial_forward=False,
same as verify_completeness.py) -- completeness is checked against the full,
unoptimized path on purpose, so a bug in the partial-forward optimization can't
silently make this check pass. Partial-forward's exactness is verified
separately by spot_check_partial.py.

Usage (mirrors args/verify_completeness.sh's grid phase, but as ONE process):
    python -m experiments.theorems.verify_completeness_grid \
        --family diffugpt --model_path models/diffugpt-m-prosqa \
        --generation_steps 12 --max_new_tokens 64 \
        --layers 0 2 4 6 8 10 12 14 16 18 20 22 \
        --check_steps 1 3 5 7 9 11 \
        --m_list 200 1000 \
        --out_dir runs/verify_completeness
"""
import gc
import argparse
import pathlib

import torch

from models.backends import build_backend
from models.model_manager import ModelManager
from attribution.hook_manager import MultiLayerHookManager
from attribution.dlig_attribution import DLIGAttribution
from experiments.theorems.verify_completeness import (
    TrajRecorder, set_seed, build_prompt_inputs,
    score_plain, score_intervene, integrate_dlig_full,
    verify_bidirectional, verify_shift,
)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Full-grid completeness check, one process")
    p.add_argument("--family", type=str, default="diffugpt", choices=["dream", "diffugpt"])
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--torch_dtype", type=str, default="float32")

    p.add_argument("--prompt", type=str, default="Explain how photosynthesis works.")
    p.add_argument("--system", type=str, default="You are a helpful assistant.")
    p.add_argument("--generation_steps", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=64)

    p.add_argument("--layers", type=str, nargs="+",
                    default=["0", "2", "4", "6", "8", "10", "12", "14", "16", "18", "20", "22"])
    p.add_argument("--check_steps", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11])
    p.add_argument("--m_list", type=int, nargs="+", default=[200, 1000])
    p.add_argument("--integration_batch_size", type=int, default=5)
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--atol", type=float, default=5e-3)
    p.add_argument("--score_mode", type=str, default="meancentered",
                    choices=["meancentered", "logprob"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="runs/verify_completeness")
    p.add_argument("--out_name", type=str, default="grid_layer_all.txt",
                    help="Must start with 'grid_layer' to match plot_completeness_grid.py's glob.")
    return p


def check_step_precomputed(dlig, real_act, baseline_act, x_t, step, layer,
                            original_length, m_list, chunk, rtol, atol):
    """Same abs/rel-error sweep as verify_completeness.check_step, but takes
    real_act/baseline_act as arguments instead of recomputing them via a
    single-layer forward pass. Prints in the SAME format so
    plot_completeness_grid.py's regexes still match."""
    print("\n" + "=" * 78)
    print(f"STEP t={step}   layer={layer}")
    print("=" * 78)

    n_unmasked = int((x_t[0, original_length:] != dlig.tokenizer.mask_token_id).sum().item()) \
        if dlig.tokenizer.mask_token_id is not None else -1
    n_gen = int(x_t.shape[1] - original_length)
    print(f"unmasked generated tokens: {n_unmasked} / {n_gen}")

    baseline_inp = dlig.create_baseline_input(
        x_t, dlig.tokenizer.mask_token_id if dlig.tokenizer.mask_token_id is not None
        else dlig.tokenizer.pad_token_id, original_length,
    )

    F_real_plain = score_plain(dlig, x_t, original_length)
    F_base_plain = score_plain(dlig, baseline_inp, original_length)
    F_real_interv = score_intervene(dlig, x_t, real_act, original_length)
    F_base_interv = score_intervene(dlig, x_t, baseline_act, original_length)

    dF = F_real_plain - F_base_plain
    denom = abs(dF) + 1e-8
    t_real = abs(F_real_plain - F_real_interv) / denom
    t_base = abs(F_base_plain - F_base_interv) / denom
    transparency_ok = (t_real < 1e-3) and (t_base < 1e-3)

    print("\n[A] hook transparency  (intervention endpoints == plain forward)")
    print(f"    F_t(X_t,C):  plain={F_real_plain:.6f}  intervene={F_real_interv:.6f}  rel_diff={t_real:.2e}")
    print(f"    F_t(X_t,/0): plain={F_base_plain:.6f}  intervene={F_base_interv:.6f}  rel_diff={t_base:.2e}")
    print(f"    deltaF = F(C) - F(/0) = {dF:.6f}")
    print(f"    -> {'PASS' if transparency_ok else 'FAIL'}")

    print("\n[B/C] completeness:  sum_{i,j} DLIG (ALL positions)  vs  deltaF")
    print(f"    {'m':>5}  {'sumDLIG':>14}  {'deltaF':>14}  {'abs_err':>11}  {'rel_err':>11}")
    last_rel = last_abs = None
    rel_by_m = []
    for m in m_list:
        dlig_raw = integrate_dlig_full(dlig, x_t, real_act, baseline_act, original_length, m=m, chunk=chunk)
        sum_dlig = float(dlig_raw.reshape(dlig_raw.shape[0], -1).sum(dim=-1).item())
        abs_err = abs(sum_dlig - dF)
        rel_err = abs_err / denom
        last_rel, last_abs = rel_err, abs_err
        rel_by_m.append(rel_err)
        print(f"    {m:>5}  {sum_dlig:>14.6f}  {dF:>14.6f}  {abs_err:>11.3e}  {rel_err:>11.3e}")
        del dlig_raw
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    completeness_ok = last_abs is not None and last_abs <= atol + rtol * abs(dF)
    print(f"    -> completeness at max m: {'PASS' if completeness_ok else 'FAIL'} "
          f"(abs_err={last_abs:.3e} vs atol+rtol*|dF|={atol + rtol * abs(dF):.3e})")

    converging = None
    if len(rel_by_m) > 1:
        converging = all(b <= a for a, b in zip(rel_by_m, rel_by_m[1:]))
        print(f"    -> convergence in m: {'PASS' if converging else 'FAIL'} "
              f"(rel_err {' -> '.join(f'{r:.3e}' for r in rel_by_m)})")

    return {"step": step, "layer": layer, "transparency_ok": transparency_ok,
            "completeness_ok": completeness_ok, "rel_err": last_rel, "abs_err": last_abs,
            "dF": dF, "converging": converging}


def main():
    args = build_arg_parser().parse_args()
    assert args.out_name.startswith("grid_layer"), \
        "--out_name must start with 'grid_layer' to match plot_completeness_grid.py's glob"
    set_seed(args.seed)

    mm = ModelManager(family=args.family, device_map=args.device_map,
                       torch_dtype=args.torch_dtype, model_path=args.model_path)
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()
    backend = build_backend(model, tokenizer, family=args.family)
    print(f"[INFO] Backend: {backend.family}  predicts_shifted={backend.predicts_shifted}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else pad_id

    input_ids, attention_mask, L = build_prompt_inputs(tokenizer, args.system, args.prompt, device)
    print(f"[INFO] prompt length L={L}")

    # --- global, layer/step-independent gates: run ONCE, not per cell ---
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    results = []
    with open(out_path, "w") as fout, _Tee(fout):
        verify_bidirectional(backend, L=8, device=device)

        rec = TrajRecorder()
        print("[INFO] Generating trajectory ONCE for the whole grid...")
        _ = backend.generate_trajectory(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens, steps=args.generation_steps,
            record_hook=rec.hook,
        )
        steps_avail = rec.steps()
        print(f"[INFO] recorded steps: {steps_avail}")
        verify_shift(backend, tokenizer, rec, L, int(mask_token_id))

        check_steps = [s for s in args.check_steps if s in steps_avail]
        if not check_steps:
            raise ValueError(f"None of --check_steps are in recorded steps {steps_avail}")

        # --- one MultiLayerHookManager for ALL analyzed layers ---
        mlhm = MultiLayerHookManager(model, layer_specs=args.layers, backend=backend)

        # One DLIGAttribution per layer (cheap: just wraps a hook view), reused across
        # steps and m -- the only per-layer object that actually needs the intervention
        # hook registered at that specific layer.
        chunk = None if args.integration_batch_size <= 0 else int(args.integration_batch_size)
        dlig_by_layer = {}
        for spec in args.layers:
            dlig_by_layer[spec] = DLIGAttribution(
                model, tokenizer, mlhm.get_layer_view(spec),
                integration_steps=max(args.m_list), integration_batch_size=chunk,
                disable_kv_cache=True, score_mode=args.score_mode,
                use_partial_forward=False,  # full-path on purpose; see module docstring
                backend=backend,
            )
            dlig_by_layer[spec].set_original_input_length(L)
            dlig_by_layer[spec].relevant_token_indices = []

        for step in check_steps:
            x_t = rec.x_by_step[step].to(device)
            baseline_inp = dlig_by_layer[args.layers[0]].create_baseline_input(
                x_t, mask_token_id, L
            )

            # TWO forward passes total for this step (real + baseline), covering
            # every analyzed layer -- vs. 2 * len(layers) in the per-layer script.
            real_acts = mlhm.capture_activations(x_t, disable_kv_cache=True)
            baseline_acts = mlhm.capture_activations(baseline_inp, disable_kv_cache=True)

            for spec in args.layers:
                r = check_step_precomputed(
                    dlig=dlig_by_layer[spec],
                    real_act=real_acts[spec].to(device),
                    baseline_act=baseline_acts[spec].to(device),
                    x_t=x_t, step=step, layer=spec, original_length=L,
                    m_list=args.m_list, chunk=chunk, rtol=args.rtol, atol=args.atol,
                )
                results.append(r)

        print("\n" + "#" * 78)
        print("VERDICT")
        print("#" * 78)
        all_pass = all(r["transparency_ok"] and r["completeness_ok"] for r in results)
        print(f"all {len(results)} cells: {'PASS' if all_pass else 'FAIL'}")
    print(f"[LOG] wrote {out_path}")


class _Tee:
    """Mirrors stdout to an already-open file handle for the duration of the block."""
    def __init__(self, fh):
        self.fh = fh
        self._orig_print = None

    def __enter__(self):
        import builtins
        self._orig_print = builtins.print

        def tee_print(*a, **kw):
            self._orig_print(*a, **kw)
            kw.pop("file", None)
            self._orig_print(*a, file=self.fh, **kw)
            self.fh.flush()

        builtins.print = tee_print
        return self

    def __exit__(self, *exc):
        import builtins
        builtins.print = self._orig_print


if __name__ == "__main__":
    main()
