#!/usr/bin/env python3
# scripts/eval_task.py
"""
eval_task.py — unified DLIG-side evaluation for WiC / ProsQA / infill on
DiffuGPT-M, for direct comparison against diffugpt/scripts/eval_task.py (the
DiffuLLaMA-repo counterpart) run with the SAME hyperparameters.

WHY THIS EXISTS
---------------
DLIG's own generation code (models/backends/diffugpt.py:generate_trajectory,
experiments/infill/attribution_infill.py:infill_generate_trajectory) is a
separate reimplementation of the official repo's random-reveal sampler
(model.py:generate_samples), so the two are not guaranteed to agree even
under identical hyperparameters. This script isolates that question: it uses
ONLY this repo's model loading (ModelManager) and generation -- never
diffugpt's own generate_samples -- so that running this file and
diffugpt/scripts/eval_task.py on the same checkpoint with matched
hyperparameters is a clean A/B test of the two generation stacks. If the two
disagree, that's a real bug in one of them, not a hyperparameter mismatch.

SEPARATOR TOKEN (WiC only)
---------------------------
WiC ddm-sft training data is formatted as
    <bos> question ====== <answer> <eos>
(see diffugpt/scripts/wic_to_diffusft.py). The shared, family-agnostic
build_prompt_inputs() in experiments/theorems/verify_completeness.py does NOT
know about this task-specific separator (it is also used by Dream and other
non-WiC experiments), so this script appends it itself for WiC -- mirroring
the fix applied in experiments/wic/{wic,wic_commitment,eval_wic}.py. Omitting
it runs the model off-distribution (per diffugpt/scripts/eval_wic.py's own
comment, this tanks accuracy).

ProsQA deliberately does NOT append the separator here (task["append_sep"] is
a no-op, _no_sep) -- experiments/prosqa/prosqa_contrastive_dlig.py's own
separator-appending code was reverted back to its original no-delimiter
behavior, and this script mirrors that so the two stay consistent. infill
runs on the BASE checkpoint (no SFT, no separator either -- see the infill
section below).

infill: reuses experiments/infill/attribution_infill.py's
infill_generate_trajectory and its oracle-span-length construction (gold
sentence's own token length), one story at a time, scored with the SAME
dependency-free word-level ROUGE-1/2/L F1 as diffugpt/scripts/eval_task.py
(not DLIG's own token-id rouge1_f1 in attribution_infill.py, and not the
`evaluate` library -- so both repos compute an identical metric with no
extra dependency).

Usage (defaults already match diffugpt/scripts/eval_task.py's defaults, so
no override needed for a matched run):
  python -m scripts.eval_task --task wic \
      --model_path models/diffugpt-m-wic --data data/wic_test_raw.jsonl

  python -m scripts.eval_task --task prosqa \
      --model_path models/diffugpt-m-prosqa --data data/prosqa_test.json

  python -m scripts.eval_task --task infill \
      --model_path models/Diffugpt --data data/rocstories_test.jsonl

If you deliberately want to compare at a DIFFERENT step count, pass the SAME
--gen_steps/--max_new_tokens here as --diffusion_steps/--gen_len to
diffugpt/scripts/eval_task.py -- never change one without the other.
"""

import os
import re
import json
import argparse

import torch
from tqdm import tqdm

from models.backends import build_backend
from models.model_manager import ModelManager
from experiments.theorems.verify_completeness import build_prompt_inputs, set_seed
from experiments.wic.wic import wic_prompt, load_wic, append_sep_token as wic_append_sep_token
from experiments.infill.attribution_infill import infill_generate_trajectory, load_stories
from experiments.prosqa.bucket_prosqa import label_row as prosqa_label_row


def _no_sep(tokenizer, input_ids, attention_mask, L):
    """ProsQA deliberately does NOT append the '======' separator, matching
    experiments/prosqa/prosqa_contrastive_dlig.py's reverted (no-delimiter)
    generation -- kept consistent so this eval script always mirrors whatever
    that attribution script actually does."""
    return input_ids, attention_mask, L


# --------------------------------------------------------------------------- #
#  Answer parsing -- deliberately mirrors diffugpt/scripts/eval_{wic,prosqa}.py
#  EXACTLY (not DLIG's own wic.py:read_pred / bucket_prosqa.py:label_row),
#  since the point of this script is a fair comparison against that parser.
# --------------------------------------------------------------------------- #
ANSWER_PREFIX = "###"


def read_yes_no(text):
    """First Yes/No in the generated text -> 1/0/None. Prefer the region after
    '###' if present; else scan the whole string. Matches
    diffugpt/scripts/eval_wic.py:read_yes_no."""
    scan = text
    if ANSWER_PREFIX in text:
        scan = text.split(ANSWER_PREFIX, 1)[1]
    t = scan.strip().lower()
    iy, ino = t.find("yes"), t.find("no")
    if iy == -1 and ino == -1:
        return None
    if iy == -1:
        return 0
    if ino == -1:
        return 1
    return 1 if iy < ino else 0


def normalize(s):
    s = s.strip().lower().rstrip(".")
    return re.sub(r"\s+", " ", s)


def parse_answer(text):
    """Matches diffugpt/scripts/eval_prosqa.py:parse_answer."""
    if ANSWER_PREFIX in text:
        after = text.split(ANSWER_PREFIX, 1)[1].strip()
        first = after.split(".", 1)[0].strip()
        return first if first else after
    sents = [s for s in re.split(r"(?<=\.)\s+", text.strip()) if s.strip()]
    return sents[-1] if sents else text.strip()


def final_concept(s):
    """Matches diffugpt/scripts/eval_prosqa.py:final_concept."""
    w = re.findall(r"[a-zA-Z]+", s)
    return w[-1].lower() if w else ""


def load_prosqa(path, n):
    data = json.load(open(path))
    return data[:n] if n > 0 else data


# --------------------------------------------------------------------------- #
#  infill: word-level ROUGE-1/2/L F1 -- IDENTICAL implementation to
#  diffugpt/scripts/eval_task.py's rouge_n_f1/rouge_l_f1, so a mismatch in
#  scores reflects the generated text, not the metric.
# --------------------------------------------------------------------------- #
from collections import Counter


def _words(s):
    return s.strip().lower().split()


def _ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def rouge_n_f1(pred, gold, n):
    p_ng, g_ng = _ngrams(_words(pred), n), _ngrams(_words(gold), n)
    overlap = sum((p_ng & g_ng).values())
    if overlap == 0 or not p_ng or not g_ng:
        return 0.0
    prec = overlap / sum(p_ng.values())
    rec = overlap / sum(g_ng.values())
    return 2 * prec * rec / (prec + rec)


def _lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(pred, gold):
    p, g = _words(pred), _words(gold)
    if not p or not g:
        return 0.0
    lcs = _lcs_len(p, g)
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


TASKS = {
    "wic": dict(
        default_data="data/wic_test_raw.jsonl",
        default_max_new_tokens=8,
        loader=lambda path, n: (load_wic(path)[:n] if n > 0 else load_wic(path)),
        prompt_fn=lambda ex: wic_prompt(ex["sentence1"], ex["sentence2"], ex["word"]),
        append_sep=wic_append_sep_token,
    ),
    "prosqa": dict(
        default_data="data/prosqa_test.json",
        default_max_new_tokens=64,
        loader=load_prosqa,
        prompt_fn=lambda ex: ex["question"].strip(),
        append_sep=_no_sep,
    ),
    "infill": dict(
        default_data="data/rocstories_test.jsonl",
        default_max_new_tokens=None,   # oracle-length span, not a fixed budget
        loader=lambda path, n: load_stories(path, n),
        prompt_fn=None,
        append_sep=None,
    ),
}


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data", default=None,
                    help="Defaults to the per-task file under data/ if omitted.")
    ap.add_argument("--n", type=int, default=-1, help="-1 => all examples.")
    ap.add_argument("--gen_steps", type=int, default=12,
                    help="Denoising steps T. DLIG's own experiment-script "
                         "default is 12; pass the SAME value to "
                         "diffugpt/scripts/eval_task.py's --diffusion_steps "
                         "for a matched comparison (e.g. 64, the train/paper "
                         "default).")
    ap.add_argument("--max_new_tokens", type=int, default=None,
                    help="Masked generation length. Defaults to the DLIG "
                         "per-task default (wic=8, prosqa=64) if omitted; "
                         "pass the SAME value to eval_task.py's --gen_len.")
    ap.add_argument("--out_file", default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def run_infill(backend, tokenizer, device, stories, gen_steps, out_file):
    """ROCStories middle-sentence infill on the BASE checkpoint. Mirrors
    experiments/infill/attribution_infill.py's tokenization and oracle span
    length (gold sentence's own token length), reusing its
    infill_generate_trajectory directly rather than re-deriving it -- but
    skips the DLIG attribution machinery entirely (just generate + score),
    and scores word-level ROUGE-1/2/L F1 (see rouge_n_f1/rouge_l_f1 above)
    instead of attribution_infill.py's own token-id rouge1_f1, so both repos
    compute the identical metric with no extra dependency."""
    mask_token_id = backend.mask_token_id()
    n, sum_r1, sum_r2, sum_rl = 0, 0.0, 0.0, 0.0
    pbar = tqdm(stories, desc="[DLIG] infill", unit="story")
    with open(out_file, "w") as fout:
        for sents in pbar:
            s1, s2, s3, s4, s5 = sents

            left_ids = tokenizer.encode(" ".join([s1, s2]), return_tensors="pt").to(device)
            gold_ids = tokenizer.encode(" " + s3, return_tensors="pt").to(device)
            right_ids = tokenizer.encode(" " + " ".join([s4, s5]), return_tensors="pt").to(device)

            span_len = gold_ids.shape[1]
            n_left = left_ids.shape[1]
            if span_len < 1 or n_left < 1 or right_ids.shape[1] < 1:
                continue

            gap_start = n_left
            gap_end = n_left + span_len
            mask_block = torch.full((1, span_len), mask_token_id,
                                    dtype=left_ids.dtype, device=device)
            x_t = torch.cat([left_ids, mask_block, right_ids], dim=1)

            with torch.no_grad():
                final_x0 = infill_generate_trajectory(
                    backend, x_t, (gap_start, gap_end),
                    steps=gen_steps, record_hook=None,
                )
            span_ids = final_x0[0, gap_start:gap_end].cpu().tolist()
            pred = tokenizer.decode(span_ids, skip_special_tokens=True).strip()
            gold = s3

            r1 = rouge_n_f1(pred, gold, 1)
            r2 = rouge_n_f1(pred, gold, 2)
            rl = rouge_l_f1(pred, gold)
            n += 1
            sum_r1 += r1; sum_r2 += r2; sum_rl += rl

            fout.write(json.dumps({
                "left": s1 + " " + s2, "right": s4 + " " + s5, "gold": gold,
                "pred": pred, "rouge1": r1, "rouge2": r2, "rougeL": rl,
            }) + "\n")
            pbar.set_postfix(rouge1=f"{100*sum_r1/n:.1f}", rougeL=f"{100*sum_rl/n:.1f}")

    print(f"\n[RESULT] task=infill  n={n}")
    print(f"  rouge1 : {100*sum_r1/n:.2f}")
    print(f"  rouge2 : {100*sum_r2/n:.2f}")
    print(f"  rougeL : {100*sum_rl/n:.2f}")
    print(f"  preds -> {out_file}")


def main():
    args = build_arg_parser().parse_args()
    task = TASKS[args.task]
    data_path = args.data or task["default_data"]
    max_new_tokens = args.max_new_tokens or task["default_max_new_tokens"]
    out_file = args.out_file or f"outputs/{args.task}/eval_task_preds.jsonl"
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    set_seed(args.seed)

    mm = ModelManager(family="diffugpt",
                      device_map=("cuda" if torch.cuda.is_available() else "cpu"),
                      torch_dtype=torch.float32,
                      model_path=args.model_path)
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()
    backend = build_backend(model, tokenizer, family="diffugpt")
    print(f"[INFO] Backend: {backend.family}  predicts_shifted={backend.predicts_shifted}")
    print(f"[INFO] task={args.task}  gen_steps={args.gen_steps}  "
          f"max_new_tokens={max_new_tokens}  data={data_path}")

    rows = task["loader"](data_path, args.n)
    print(f"[INFO] {len(rows)} examples")

    if args.task == "infill":
        run_infill(backend, tokenizer, device, rows, args.gen_steps, out_file)
        return

    n = n_correct = n_exact = n_concept = n_unreadable = 0
    per_class = {0: [0, 0], 1: [0, 0]}
    bucket_counts = {"correct": 0, "concept_only": 0, "wrong_valid": 0,
                      "concept_invalid": 0, "subj_wrong": 0}
    pbar = tqdm(rows, desc=f"[DLIG] {args.task}", unit="ex")
    with open(out_file, "w") as fout:
        for row in pbar:
            prompt = task["prompt_fn"](row)
            input_ids, attention_mask, L = build_prompt_inputs(
                tokenizer, "", prompt, device
            )
            input_ids, attention_mask, L = task["append_sep"](
                tokenizer, input_ids, attention_mask, L
            )

            x0 = backend.generate_trajectory(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, steps=args.gen_steps,
                record_hook=None,
            )
            gen_ids = x0[0][L:].tolist()
            eos_id = tokenizer.eos_token_id
            if eos_id is not None and eos_id in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(eos_id)]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            n += 1
            if args.task == "wic":
                pred = read_yes_no(gen_text)
                gold = row["label"]
                ok = (pred is not None and pred == gold)
                per_class[gold][1] += 1
                if pred is None:
                    n_unreadable += 1
                else:
                    n_correct += int(ok)
                    per_class[gold][0] += int(ok)
                rec = {"prompt": prompt, "label": gold, "gen": gen_text,
                       "pred": pred, "ok": bool(ok)}
                pbar.set_postfix(acc=f"{100*n_correct/n:.1f}%")
            else:
                pred = parse_answer(gen_text)
                gold = row["answer"].strip()
                exact = normalize(pred) == normalize(gold)
                concept = final_concept(pred) == final_concept(gold)
                n_exact += int(exact)
                n_concept += int(concept)
                rec = {"question": prompt, "gold": gold, "gen": gen_text,
                       "pred_answer": pred, "exact": exact, "concept_match": concept}
                # bucket_prosqa.py's success metric: exact/concept_match gated
                # by whether the model even answered about the right entity
                # (raw concept_match alone can match on a WRONG subject).
                bucket = prosqa_label_row(rec, n)["bucket"]
                bucket_counts[bucket] += 1
                rec["bucket"] = bucket
                success = bucket_counts["correct"] + bucket_counts["concept_only"]
                pbar.set_postfix(exact=f"{100*n_exact/n:.1f}%",
                                 success=f"{100*success/n:.1f}%")

            fout.write(json.dumps(rec) + "\n")

    print(f"\n[RESULT] task={args.task}  n={n}")
    if args.task == "wic":
        acc = n_correct / n if n else 0.0
        a1 = per_class[1][0] / per_class[1][1] if per_class[1][1] else 0.0
        a0 = per_class[0][0] / per_class[0][1] if per_class[0][1] else 0.0
        print(f"  accuracy        : {100*acc:.2f}%   (unreadable counted wrong: {n_unreadable})")
        print(f"  same-sense (1)  : {100*a1:.2f}%   ({per_class[1][0]}/{per_class[1][1]})")
        print(f"  diff-sense (0)  : {100*a0:.2f}%   ({per_class[0][0]}/{per_class[0][1]})")
    else:
        success = bucket_counts["correct"] + bucket_counts["concept_only"]
        fail = bucket_counts["wrong_valid"]
        off = bucket_counts["subj_wrong"] + bucket_counts["concept_invalid"]
        print(f"  answer_exact     : {100*n_exact/n:.2f}%   (unguarded string match, "
              f"kept for reference)")
        print(f"  raw concept_match: {100*n_concept/n:.2f}%   (unguarded -- can match "
              f"on the WRONG subject; not the metric to report)")
        for b, c in bucket_counts.items():
            print(f"  bucket {b:16}: {c:4}  ({100*c/n:5.1f}%)")
        print(f"  SUCCESS (correct + concept_only, subject-gated): "
              f"{100*success/n:.2f}%   ({success}/{n})  <-- the metric to report")
        print(f"  fail   (wrong_valid)                          : {100*fail/n:.2f}%   ({fail}/{n})")
        print(f"  off-manifold (subj_wrong + concept_invalid)   : {100*off/n:.2f}%   ({off}/{n})")
    print(f"  preds -> {out_file}")


if __name__ == "__main__":
    main()
