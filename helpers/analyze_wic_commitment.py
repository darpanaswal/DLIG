#!/usr/bin/env python3
# helpers/analyze_wic_commitment.py
"""
Analyse the denoising-step commitment probe (wic_commitment.jsonl).

Per example we have P(Yes) at the answer slot across denoising steps. We reduce
each trajectory to WHEN it committed and ask whether bias-defaulted Yes commits
earlier than genuine answers — a population comparison, no per-example label.

Two commitment scalars, both step-index based (0 = first/most-noised step,
larger = later/cleaner):
  commit_step : first step after which p_yes_2way stays on the committed side
                (>0.5 for a Yes commit, <0.5 for a No commit) through the end.
  auc_early   : mean p_yes_2way over the FIRST HALF of steps — how Yes-leaning
                the model is while the canvas is still noisy. High + a Yes commit
                = locked in early.

Contrasts (predicted-Yes only, so the committed answer is the same):
  genuine  = pred Yes & gold Yes
  defaulted= pred Yes & gold No   (the bias failures)
If defaulted commits earlier (smaller commit_step / higher auc_early), that's the
shortcut signature on the time axis.
"""

import json
import argparse
import numpy as np


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def commit_step(p2, side):
    """First index from which p2 stays on `side` (>0.5 if Yes-commit else <0.5)
    through the end. Returns len(p2) if it never stabilises (late/never)."""
    p2 = np.asarray(p2, float)
    n = len(p2)
    on = (p2 > 0.5) if side == 1 else (p2 < 0.5)
    stable = n
    for i in range(n):
        if on[i] and on[i:].all():
            stable = i
            break
    return stable / max(n - 1, 1)  # normalise to 0..1


def auc_mw(pos, neg):
    pos = np.asarray([x for x in pos if not np.isnan(x)], float)
    neg = np.asarray([x for x in neg if not np.isnan(x)], float)
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, (n1, n2)
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    tie = np.zeros(len(cnt)); cum = 0
    for k, c in enumerate(cnt):
        tie[k] = cum + (c + 1) / 2.0; cum += c
    ranks = tie[inv]
    U1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    auc = U1 / (n1 * n2)
    mu = n1 * n2 / 2.0
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    from math import erfc, sqrt
    p = erfc(abs((U1 - mu) / (sd + 1e-12)) / sqrt(2))
    return float(auc), float(p), (n1, n2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default="outputs/wic/wic_commitment.jsonl")
    args = ap.parse_args()
    rows = load(args.in_file)

    for r in rows:
        p2 = r["p_yes_2way"]
        side = r["pred"] if r["pred"] in (0, 1) else (1 if p2[-1] > 0.5 else 0)
        r["commit_step"] = commit_step(p2, side)
        half = max(1, len(p2) // 2)
        r["auc_early"] = float(np.mean(p2[:half]))     # early Yes-leaning
        r["p_yes_final"] = float(p2[-1])

    def grp(pred, label):
        return [r for r in rows if r["pred"] == pred and r["label"] == label]

    gen = grp(1, 1)      # genuine Yes
    dfl = grp(1, 0)      # defaulted Yes (bias failures)
    cn  = grp(0, 0)      # correct No
    print(f"predicted-Yes: genuine(goldYes) n={len(gen)}  "
          f"defaulted(goldNo) n={len(dfl)}   correct-No n={len(cn)}")

    def report(name, A, B, key, higher_means):
        a = [r[key] for r in A]; b = [r[key] for r in B]
        auc, p, (n1, n2) = auc_mw(a, b)
        print(f"\n[{name}] {key}")
        print(f"   mean A={np.nanmean(a):.3f}  mean B={np.nanmean(b):.3f}  "
              f"AUC={auc:.3f}  p={p:.4f}")
        print(f"   (AUC>.5 => A tends higher; {higher_means})")

    print("\n=== genuine Yes (A) vs defaulted Yes (B) ===")
    report("commit-timing", gen, dfl, "commit_step",
           "higher commit_step = commits LATER")
    report("early-lean", gen, dfl, "auc_early",
           "higher = more Yes-leaning while still noisy")

    print("\n=== correct-No (A) vs defaulted Yes (B) ===")
    report("commit-timing", cn, dfl, "commit_step",
           "higher commit_step = commits LATER")

    print("\nInterpretation: if defaulted Yes has SMALLER commit_step and HIGHER "
          "auc_early than genuine Yes / correct-No, the bias locks in early on the "
          "denoising axis — the shortcut signature. Population claim only.")


if __name__ == "__main__":
    main()