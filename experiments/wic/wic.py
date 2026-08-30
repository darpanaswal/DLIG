#!/usr/bin/env python3
# experiments/wic.py
"""
wic.py — self-generated DLIG attribution on WiC-finetuned DiffuGPT-M, for
the QUALITATIVE / VISUALIZATION demonstration (no aggregated claims, no
token-role labels — by design).

SELF-GENERATED TARGET (like the infill experiment, NOT contrastive):
the model denoises its own answer ("### Yes" / "### No"); DLIG then attributes
the model's commitment to WHATEVER IT GENERATED back onto the prompt tokens.
There is no y+/y- contrast and no fixed gold target — the committed answer IS
the target. Mechanically: leave dlig.target_output_ids = None, which selects
DLIGAttribution's self-generated scoring path (it reads the committed tokens at
the answer positions and applies the predicts_shifted offset automatically).

Per example we attribute over all prompt tokens, at each requested denoising
step and layer. The plotter averages over steps for the main per-token/per-layer
bar panels, and can also show one example's per-step evolution (no claim).

Carried through for plot-time filtering only (NOT used to compute attribution):
gold label, the model's own prediction, correctness, and the decoded answer.

Output: resumable JSONL keyed by idx, one record per example:
    {idx, word, label, pred, correct, gen_text,
     input_tokens,                       # kept prompt tokens
     steps_data: [{step, layers: {l: [score per prompt token]}}]}
"""

import os
import gc
import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

from models.backends import build_backend
from utils.config import OUTPUT_DIR
from attribution.dlig_attribution import DLIGAttribution
from attribution.hook_manager import MultiLayerHookManager
from models.model_manager import ModelManager
from experiments.theorems.verify_completeness import (
    set_seed, build_prompt_inputs,
)
from experiments.contrastive.contrastive_attribution import clean_token, input_token_indices


def wic_prompt(sentence1: str, sentence2: str, word: str) -> str:
    """Must match the TRAINING prompt (wic_to_diffusft.py build_prompt)."""
    return (
        f'Sentence 1: {sentence1}\n'
        f'Sentence 2: {sentence2}\n'
        f'Does the word "{word}" have the same meaning in both sentences?'
    )


def read_pred(text):
    """First Yes/No in generated text -> 1/0/None."""
    t = text.strip().lower()
    iy, ino = t.find("yes"), t.find("no")
    if iy == -1 and ino == -1:
        return None
    if iy == -1:
        return 0
    if ino == -1:
        return 1
    return 1 if iy < ino else 0


def load_wic(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({
                    "sentence1": r["sentence1"],
                    "sentence2": r["sentence2"],
                    "word": r.get("word") or r.get("lemma"),
                    "label": int(r["label"]),
                })
    return rows


def build_arg_parser():
    p = argparse.ArgumentParser(description="WiC self-generated DLIG (qualitative)")
    p.add_argument("--wic_jsonl", type=str, default="data/wic_test_raw.jsonl")
    p.add_argument("--out_file", type=str,
                   default=str(OUTPUT_DIR / "wic/wic_dlig.jsonl"))
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--n", type=int, default=-1, help="-1 => all rows")
    p.add_argument("--system", type=str, default="")

    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)

    # DLIG hyperparameters (paper setting: T=12, steps {1,3,5,7,9,11}, even layers)
    p.add_argument("--m", type=int, default=12)
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--gen_steps", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=6)
    p.add_argument("--target_steps", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11])
    p.add_argument("--layers", type=str, nargs="+",
                   default=[str(i) for i in range(0, 24, 2)])
    p.add_argument("--score_mode", type=str, default="meancentered")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    if args.num_shards > 1:
        op = Path(args.out_file)
        combined_file = args.out_file
        args.out_file = str(op.parent / f"{op.stem}_shard{args.shard_id}{op.suffix}")
    else:
        combined_file = None
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    Path(args.out_file).touch(exist_ok=True)

    rows = load_wic(args.wic_jsonl)
    if args.n >= 0:
        rows = rows[:args.n]

    mm = ModelManager(family="diffugpt",
                      device_map=("cuda" if torch.cuda.is_available() else "cpu"),
                      torch_dtype=torch.float32,
                      model_path=args.model_path)
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()

    backend = build_backend(model, tokenizer, family="diffugpt")
    print(f"[INFO] Backend: {backend.family}  predicts_shifted={backend.predicts_shifted}")

    mask_token_id = backend.mask_token_id()
    n_layers = backend.num_layers()
    valid_layers = [l for l in args.layers if int(l) < n_layers]
    mlhm = MultiLayerHookManager(model, layer_specs=valid_layers, backend=backend)

    dlig = DLIGAttribution(
        model, tokenizer, mlhm.get_layer_view(valid_layers[0]),
        integration_steps=args.m, integration_batch_size=args.chunk,
        disable_kv_cache=True, score_mode=args.score_mode,
        use_partial_forward=True,
        backend=backend,
    )
    # SELF-GENERATED: ensure no fixed target is set, so DLIG scores the model's
    # own committed answer tokens (its self-generated path).
    dlig.target_output_ids = None

    indexed = list(enumerate(rows))
    if args.num_shards > 1:
        indexed = indexed[args.shard_id :: args.num_shards]
    print(f"[INFO Shard {args.shard_id}/{args.num_shards}] {len(indexed)} candidate "
          f"examples. Output -> {args.out_file}")

    done = set()
    for rf in filter(None, [args.out_file, combined_file]):
        if os.path.exists(rf):
            with open(rf) as f:
                for line in f:
                    if line.strip():
                        done.add(json.loads(line)["idx"])
    if done:
        print(f"[INFO] Resuming past {len(done)} examples.")

    for idx, r in tqdm(indexed, desc=f"shard {args.shard_id}"):
        if idx in done:
            continue

        prompt = wic_prompt(r["sentence1"], r["sentence2"], r["word"])
        input_ids, attention_mask, L = build_prompt_inputs(
            tokenizer, args.system, prompt, device
        )

        # one denoising trajectory; record x at every step, and the committed x0
        rec = SimpleNamespace(x_by_step={})
        def _rec_hook(step, xt_cpu, logits):
            rec.x_by_step[int(step)] = xt_cpu
        x0 = backend.generate_trajectory(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens, steps=args.gen_steps,
            record_hook=_rec_hook,
        )
        gen_ids = x0[0][L:].tolist()
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(eos_id)]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred = read_pred(gen_text)
        correct = (pred is not None and pred == r["label"])
        # store EVERY example (correct, incorrect, and unparseable pred=None):
        # failure modes of the Yes-biased model are the point of this analysis.

        keep_idx = input_token_indices(input_ids[0, :L].tolist(), tokenizer,
                                       user_prompt=prompt)
        prompt_tokens = [clean_token(t)
                         for t in tokenizer.convert_ids_to_tokens(input_ids[0, :L])]
        kept_tokens = [prompt_tokens[i] for i in keep_idx]

        # attribute over all prompt positions (0..L-1); self-generated target
        dlig.set_original_input_length(L)
        dlig.relevant_token_indices = list(range(L))

        out = {
            "idx": idx, "word": r["word"], "label": r["label"],
            "pred": pred, "correct": bool(correct), "gen_text": gen_text,
            "input_tokens": kept_tokens,
            "steps_data": [],
        }

        for step in args.target_steps:
            if step not in rec.x_by_step:
                continue
            x_step = rec.x_by_step[step].to(device)

            # baseline: mask the whole prompt, keep the answer-span state
            baseline_step = x_step.clone()
            baseline_step[:, :L] = mask_token_id

            with torch.no_grad():
                real_acts = mlhm.capture_activations(x_step, disable_kv_cache=True)
                baseline_acts = mlhm.capture_activations(baseline_step,
                                                         disable_kv_cache=True)

            step_data = {"step": step, "layers": {}}
            for layer in valid_layers:
                dlig.hook_manager = mlhm.get_layer_view(layer)
                res = dlig.compute_dlig_at_timestep_with_activations(
                    step=step, x_t=x_step,
                    real_act=real_acts[layer], baseline_act=baseline_acts[layer],
                    original_length=L,
                )
                # per-position score s[i] = sum_j DLIG[i, j]; keep prompt tokens
                s = res["full_dlig"][0].sum(dim=-1).float().cpu().numpy()
                step_data["layers"][layer] = s[keep_idx].tolist()

            out["steps_data"].append(step_data)

        with open(args.out_file, "a") as f:
            f.write(json.dumps(out) + "\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()