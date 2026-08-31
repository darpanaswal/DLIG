#!/usr/bin/env python3
# experiments/wic/eval_wic.py
"""
Dead-simple WiC eval. Generate the answer, read the first Yes/No, compare to label.
No forced-choice, no logit reading, no masked-slot tricks.

Usage:
  python -u -m experiments.wic.eval_wic \
      --model_path models/diffugpt-m-wic \
      --wic_jsonl data/wic_test_raw.jsonl
"""

import os
import json
import argparse
import torch

from models.backends import build_backend
from models.model_manager import ModelManager
from experiments.theorems.verify_completeness import (
    set_seed, build_prompt_inputs,
)
from experiments.wic.wic import append_sep_token


def wic_prompt(sentence1, sentence2, word):
    # Match the TRAINING prompt format exactly (from wic_to_diffusft.py build_prompt).
    return (
        f'Sentence 1: {sentence1}\n'
        f'Sentence 2: {sentence2}\n'
        f'Does the word "{word}" have the same meaning in both sentences?'
    )


def read_answer(text):
    """First Yes/No in the generated text. Returns 1 (yes), 0 (no), or None."""
    t = text.strip().lower()
    # find whichever of yes/no appears first
    iy = t.find("yes")
    ino = t.find("no")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--wic_jsonl", default="data/wic_test_raw.jsonl")
    ap.add_argument("--out_file", default="outputs/wic/eval_simple.json")
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--gen_steps", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

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

    correct = 0
    readable = 0
    per_example = []

    for r in rows:
        prompt = wic_prompt(r["sentence1"], r["sentence2"], r["word"])
        input_ids, attention_mask, L = build_prompt_inputs(tokenizer, "", prompt, device)
        input_ids, attention_mask, L = append_sep_token(
            tokenizer, input_ids, attention_mask, L
        )

        x0 = backend.generate_trajectory(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens, steps=args.gen_steps,
            record_hook=None,   # skip the per-step CPU trajectory copy; we only need the final x0
        )
        gen_text = tokenizer.decode(x0[0][L:].tolist(), skip_special_tokens=True).strip()
        pred = read_answer(gen_text)

        ok = 0
        if pred is not None:
            readable += 1
            ok = int(pred == r["label"])
            correct += ok
        per_example.append({"word": r["word"], "label": r["label"],
                            "pred": pred, "gen_text": gen_text, "ok": ok})

    n = len(rows)
    acc_all = correct / n                              # unreadable counted wrong
    acc_readable = correct / readable if readable else 0.0

    summary = {
        "model_path": args.model_path,
        "n": n,
        "accuracy": acc_all,
        "accuracy_over_readable": acc_readable,
        "readable": readable,
        "correct": correct,
    }
    with open(args.out_file, "w") as f:
        json.dump({"summary": summary, "per_example": per_example}, f, indent=2)

    print(f"\n  n              : {n}")
    print(f"  readable       : {readable}/{n}")
    print(f"  accuracy       : {acc_all:.3f}  (unreadable = wrong)")
    print(f"  acc (readable) : {acc_readable:.3f}")
    print(f"  chance         : 0.500")
    print(f"[INFO] wrote {args.out_file}")


if __name__ == "__main__":
    main()