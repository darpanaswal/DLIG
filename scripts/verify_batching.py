#!/usr/bin/env python3
# scripts/verify_batching.py
"""
Verifies that batched generation matches single-example generation on the
ACTUAL trained checkpoint, for WiC and ProsQA (the two tasks with batching
wired into their generation loops -- see experiments/wic/wic.py and
experiments/prosqa/prosqa_contrastive_dlig.py).

This is the real-weights counterpart to the random-init mechanism test used
to design the fix (models/backends/diffugpt.py's padding-aware attn mask +
position ids). That test proved the mechanism is correct independent of
weights; this script confirms it holds for your actual checkpoint too.

Method: take N examples, run each ALONE (no padding) to get ground-truth
step-0 logits at every real (non-mask) position, then run all N together as
one left-padded batch and compare. Comparing pre-sampling LOGITS rather than
generated text is deliberate -- categorical sampling draws randomness
differently for batched vs sequential calls even with the same seed, so
generated tokens are not expected to match token-for-token; the forward pass
itself (embeddings + attention over the correct, unpadded context) is what
must match.

Usage:
  python -m scripts.verify_batching --task wic \
      --model_path models/diffugpt-m-wic --data data/wic_test_raw.jsonl --n 6

  python -m scripts.verify_batching --task prosqa \
      --model_path models/diffugpt-m-prosqa --data data/prosqa_test.json --n 6
"""
import json
import argparse

import torch

from models.backends import build_backend
from models.model_manager import ModelManager
from experiments.theorems.verify_completeness import build_prompt_inputs, set_seed
from experiments.wic.wic import wic_prompt, load_wic, append_sep_token as wic_append_sep_token
from experiments.prosqa.prosqa_contrastive_dlig import (
    append_sep_token as prosqa_append_sep_token,
)


def build_examples(task, data_path, n):
    if task == "wic":
        rows = load_wic(data_path)[:n]
        return [wic_prompt(r["sentence1"], r["sentence2"], r["word"]) for r in rows]
    else:
        data = json.load(open(data_path))[:n]
        return [ex["question"].strip() for ex in data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["wic", "prosqa"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=6, help="Number of examples in the test batch.")
    ap.add_argument("--gen_steps", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable_tf32", action="store_true",
                    help="No longer needed for correctness -- model_manager.py "
                         "no longer enables TF32 by default (confirmed via this "
                         "script that it corrupts batched generation by up to "
                         "~0.8 in logit space). Kept as an explicit override in "
                         "case TF32 is re-enabled or torch's global default "
                         "changes; forces full fp32 matmul precision.")
    args = ap.parse_args()

    data_path = args.data or ("data/wic_test_raw.jsonl" if args.task == "wic"
                               else "data/prosqa_test.json")
    max_new_tokens = args.max_new_tokens or (8 if args.task == "wic" else 64)
    append_sep = wic_append_sep_token if args.task == "wic" else prosqa_append_sep_token

    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("[INFO] TF32 forced OFF (full fp32 matmul precision)")

    set_seed(args.seed)
    mm = ModelManager(family="diffugpt",
                      device_map=("cuda" if torch.cuda.is_available() else "cpu"),
                      torch_dtype=torch.float32,
                      model_path=args.model_path)
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()
    backend = build_backend(model, tokenizer, family="diffugpt")
    print(f"[INFO] task={args.task}  device={device}  gen_steps={args.gen_steps}  "
          f"max_new_tokens={max_new_tokens}")

    prompts = build_examples(args.task, data_path, args.n)
    print(f"[INFO] {len(prompts)} examples")

    # --- build each example's (input_ids, attention_mask, L), with the sep --- #
    encs = []
    for p in prompts:
        ids, am, L = build_prompt_inputs(tokenizer, "", p, device)
        ids, am, L = append_sep(tokenizer, ids, am, L)
        encs.append((ids, am, L))

    # --- ALONE: run each example separately, capture step-0 logits --- #
    alone_logits = []
    for ids, am, L in encs:
        captured = {}
        def hook(step, xt, logits, _c=captured):
            if step == 0:
                _c["logits"] = logits.clone()
        backend.generate_trajectory(ids, attention_mask=am, max_new_tokens=max_new_tokens,
                                    steps=args.gen_steps, record_hook=hook)
        alone_logits.append(captured["logits"][0])   # [L+max_new_tokens, |V|]

    # --- BATCHED: left-pad all examples together, one call --- #
    Lmax = max(L for _, _, L in encs)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids_rows, mask_rows, lens = [], [], []
    for ids, am, L in encs:
        n_pad = Lmax - L
        pad_ids = torch.full((1, n_pad), pad_id, dtype=ids.dtype, device=device)
        pad_mask = torch.zeros((1, n_pad), dtype=am.dtype, device=device)
        ids_rows.append(torch.cat([pad_ids, ids], dim=1))
        mask_rows.append(torch.cat([pad_mask, am], dim=1))
        lens.append(L)
    batch_ids = torch.cat(ids_rows, dim=0)
    batch_mask = torch.cat(mask_rows, dim=0)

    captured_batch = {}
    def batch_hook(step, xt, logits):
        if step == 0:
            captured_batch["logits"] = logits.clone()
    backend.generate_trajectory(batch_ids, attention_mask=batch_mask,
                                max_new_tokens=max_new_tokens, steps=args.gen_steps,
                                record_hook=batch_hook)
    batch_logits = captured_batch["logits"]  # [B, Lmax+max_new_tokens, |V|]

    # --- compare, at each example's real (non-pad) positions --- #
    print(f"\n[RESULT] step-0 logits, batched vs alone (max abs diff per example):")
    worst = 0.0
    for i, L in enumerate(lens):
        n_pad = Lmax - L
        d = (batch_logits[i, n_pad:] - alone_logits[i]).abs().max().item()
        worst = max(worst, d)
        flag = "OK" if d < 1e-2 else "MISMATCH"
        print(f"  example {i} (L={L}, pad={n_pad:3d}): diff={d:.4e}  [{flag}]")

    print(f"\n[SUMMARY] worst-case diff across {len(lens)} examples: {worst:.4e}")
    if worst < 1e-2:
        print("PASS -- batched generation matches single-example generation "
              "(diffs are ordinary fp32 GPU accumulation noise -- calibrated "
              "against a real checkpoint at ~1e-3 to 1e-4, NOT a real "
              "divergence, which showed up two-to-three orders of magnitude "
              "larger, ~0.1-0.8, when TF32 was enabled -- see model_manager.py).")
    else:
        print("FAIL -- batched and single-example generation disagree beyond "
              "float noise. Do not trust batched results until this is "
              "resolved.")


if __name__ == "__main__":
    main()
