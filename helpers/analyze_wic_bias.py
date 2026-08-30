#!/usr/bin/env python3
# helpers/analyze_wic_bias.py
"""
analyze_wic_bias.py — two independent, read-only diagnostics for the WiC
"yes-bias" investigation (does the model discriminate senses, or just default
to one answer?). Neither diagnostic runs the model; both just inspect files
already on disk. Each runs on its own if its input file is given/found — you
don't need both.

DIAGNOSIS 1 — EVAL-TIME bias (--eval_json, from eval_wic.py's eval.json)
  Is the model's raw accuracy real discrimination, or an artifact of always
  guessing the majority class? Reports:
    - overall accuracy (readable + over-all, unreadable counted wrong)
    - per-class accuracy (label 1 = same sense, label 0 = different)
    - BALANCED accuracy = mean of the two per-class accuracies — the honest
      number; a constant-"Yes" predictor scores ~0.5 here even if raw
      accuracy looks high
    - prediction distribution (Yes-rate) vs label distribution
    - confusion matrix
    - a verdict bucketing balanced accuracy into chance / weak / real
      discrimination

DIAGNOSIS 2 — TRAIN-TIME bias source (--train_jsonl, from wic_train.jsonl)
  If Diagnosis 1 finds a bias, where does it enter? Checks:
    1. Label balance in the training set (should be ~50/50 Yes/No).
    2. Whether "### Yes." and "### No." tokenize to the same length — an
       uneven target length would give the diffusion loss an easier class.
    3. Whether <|endoftext|> maps to the tokenizer's actual EOS id (50256)
       rather than being written as literal text — if it isn't, the model
       never sees a real EOS to learn to stop on.

Usage:
  python -m helpers.analyze_wic_bias --eval_json outputs/wic/eval.json
  python -m helpers.analyze_wic_bias --train_jsonl data/wic_train.jsonl
  python -m helpers.analyze_wic_bias \\
      --eval_json outputs/wic/eval.json --train_jsonl data/wic_train.jsonl
"""

import os
import json
import argparse
from collections import Counter


# ============================================================ diagnosis 1: eval-time bias

def diagnose_eval(eval_json):
    d = json.load(open(eval_json))
    pe = d["per_example"]
    n = len(pe)

    lab1 = [e for e in pe if e["label"] == 1]
    lab0 = [e for e in pe if e["label"] == 0]

    def acc(rows):
        r = [e for e in rows if e.get("pred") is not None]
        if not r:
            return float("nan"), 0, 0
        c = sum(int(e["pred"] == e["label"]) for e in r)
        return c / len(r), c, len(r)

    a1, c1, r1 = acc(lab1)   # accuracy on same-sense (correct = predict Yes)
    a0, c0, r0 = acc(lab0)   # accuracy on different-sense (correct = predict No)
    a_all, c_all, r_all = acc(pe)

    readable = [e for e in pe if e.get("pred") is not None]
    pred_yes = sum(1 for e in readable if e["pred"] == 1)
    pred_no = sum(1 for e in readable if e["pred"] == 0)
    unreadable = n - len(readable)

    def cnt(lab, pred):
        return sum(1 for e in readable if e["label"] == lab and e["pred"] == pred)
    tp = cnt(1, 1); fn = cnt(1, 0)   # label 1
    fp = cnt(0, 1); tn = cnt(0, 0)   # label 0

    balanced = (a1 + a0) / 2 if (r1 and r0) else float("nan")

    print("\n" + "=" * 60)
    print("DIAGNOSIS 1 — eval-time bias")
    print("=" * 60)
    print(f"n total            : {n}")
    print(f"readable           : {len(readable)}  (unreadable {unreadable})")
    print(f"label dist         : same(1)={len(lab1)}  diff(0)={len(lab0)}")
    print(f"pred  dist (read.) : Yes={pred_yes}  No={pred_no}"
          f"   (Yes-rate {pred_yes/max(len(readable),1):.3f})")
    print("-" * 44)
    print(f"acc over-all       : {c_all}/{n} = {c_all/n:.3f}  (unreadable=wrong)")
    print(f"acc readable       : {a_all:.3f}  ({c_all}/{r_all})")
    print(f"acc same-sense (1) : {a1:.3f}  ({c1}/{r1})   correct = says Yes")
    print(f"acc diff-sense (0) : {a0:.3f}  ({c0}/{r0})   correct = says No")
    print("-" * 44)
    print(f"BALANCED accuracy  : {balanced:.3f}   <-- the honest gate number")
    print(f"                     (constant-Yes predictor scores ~0.50 here)")
    print("-" * 44)
    print("confusion (readable):")
    print(f"              pred Yes   pred No")
    print(f"  true same     {tp:5d}     {fn:5d}")
    print(f"  true diff     {fp:5d}     {tn:5d}")
    print("-" * 44)
    if balanced == balanced:  # not nan
        if balanced < 0.53:
            print("VERDICT: at/near chance after de-biasing. Model is NOT")
            print("         discriminating senses — raw acc is a bias artifact.")
        elif balanced < 0.60:
            print("VERDICT: weak but non-zero discrimination. Bias correction")
            print("         might surface a little more; ceiling is low.")
        else:
            print("VERDICT: real discrimination present. Bias correction worth it.")


# ============================================================ diagnosis 2: train-time bias source

def diagnose_train(train_jsonl):
    rows = [json.loads(l) for l in open(train_jsonl) if l.strip()]
    n = len(rows)

    print("\n" + "=" * 60)
    print("DIAGNOSIS 2 — train-time bias source")
    print("=" * 60)

    # 1. label balance, inferred from the output text (### Yes / ### No)
    ans = Counter()
    for r in rows:
        o = r["output"]
        if "Yes" in o:
            ans["Yes"] += 1
        elif "No" in o:
            ans["No"] += 1
        else:
            ans["?"] += 1
    print(f"n train            : {n}")
    print(f"answer balance     : {dict(ans)}")
    print(f"  Yes-rate         : {ans['Yes']/n:.3f}  (want ~0.500)")
    print("-" * 44)

    # 2 + 3. tokenization of the two answer targets
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"[WARN] tokenizer unavailable ({e}); skipping token inspection.")
        return

    eos_id = tok.eos_token_id
    print(f"gpt2 eos id        : {eos_id}")
    for out in ['### Yes.<|endoftext|>', '### No.<|endoftext|>']:
        ids = tok(out, add_special_tokens=False)["input_ids"]
        toks = [tok.decode([t]) for t in ids]
        has_eos = eos_id in ids
        print(f"\n  target {out!r}")
        print(f"    ids   : {ids}")
        print(f"    toks  : {toks}")
        print(f"    len   : {len(ids)}   eos present as id: {has_eos}")
        if not has_eos:
            print(f"    [!] <|endoftext|> did NOT tokenize to the eos id — the model")
            print(f"        never sees a real EOS, so it can't learn to stop.")

    # length parity check
    y = tok('### Yes.<|endoftext|>', add_special_tokens=False)["input_ids"]
    nn = tok('### No.<|endoftext|>', add_special_tokens=False)["input_ids"]
    print("-" * 44)
    if len(y) != len(nn):
        print(f"[!] Yes target len ({len(y)}) != No target len ({len(nn)}).")
        print("    Unequal answer-span lengths give the diffusion loss an uneven")
        print("    target across classes and can drive a bias. Consider padding")
        print("    the shorter to match, or a single-token answer scheme.")
    else:
        print(f"[OK] Yes/No targets same length ({len(y)} tokens); differ only in")
        print("     the answer token — no length-induced class bias.")


# ============================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", default="outputs/wic/eval.json",
                    help="Diagnosis 1 input (eval_wic.py output); skipped if absent")
    ap.add_argument("--train_jsonl", default="data/wic_train.jsonl",
                    help="Diagnosis 2 input; skipped if absent")
    args = ap.parse_args()

    ran_any = False
    if os.path.exists(args.eval_json):
        diagnose_eval(args.eval_json)
        ran_any = True
    else:
        print(f"[skip diagnosis 1] {args.eval_json} not found")

    if os.path.exists(args.train_jsonl):
        diagnose_train(args.train_jsonl)
        ran_any = True
    else:
        print(f"[skip diagnosis 2] {args.train_jsonl} not found")

    if not ran_any:
        raise SystemExit("Neither --eval_json nor --train_jsonl exists; nothing to diagnose.")


if __name__ == "__main__":
    main()
