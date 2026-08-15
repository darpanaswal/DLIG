#!/usr/bin/env python3
# diagnose_wic_bias.py
"""
Diagnose WiC eval output: is the model discriminating senses, or just biased
toward one answer? Reads the eval.json written by eval_wic.py.

Reports:
  - overall accuracy (readable + over-all)
  - per-class accuracy (label 1 = same sense, label 0 = different)
  - BALANCED accuracy = mean of the two per-class accuracies (the honest number;
    a constant-"Yes" predictor scores ~0.5 here even if raw acc looks higher)
  - prediction distribution (how often it says Yes vs No) vs label distribution
  - confusion matrix

Usage:
  python diagnose.py --eval_json outputs/wic/eval.json
"""

import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", default="outputs/wic/eval.json")
    args = ap.parse_args()

    d = json.load(open(args.eval_json))
    pe = d["per_example"]
    n = len(pe)

    # counts
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

    # prediction distribution among readable
    readable = [e for e in pe if e.get("pred") is not None]
    pred_yes = sum(1 for e in readable if e["pred"] == 1)
    pred_no = sum(1 for e in readable if e["pred"] == 0)
    unreadable = n - len(readable)

    # confusion (readable only)
    def cnt(lab, pred):
        return sum(1 for e in readable if e["label"] == lab and e["pred"] == pred)
    tp = cnt(1, 1); fn = cnt(1, 0)   # label 1
    fp = cnt(0, 1); tn = cnt(0, 0)   # label 0

    balanced = (a1 + a0) / 2 if (r1 and r0) else float("nan")

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


if __name__ == "__main__":
    main()