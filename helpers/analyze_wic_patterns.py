#!/usr/bin/env python3
# helpers/analyze_wic_patterns.py
"""
analyze_wic_patterns.py — rule-free pattern search over WiC DLIG attribution.

Derive per-example scalar features purely from the single-target self-generated
DLIG tensor (same signal plot_wic.py draws, via signed_d_by_layer), then test
which features separate the populations (different-sense vs same-sense; No vs Yes;
correct vs incorrect) with univariate AUC + Mann-Whitney. No hand-defined token
roles: the pivot is read from row["word"], the sentence boundary from the
delimiter token; everything else is attribution-mass geometry.

Reports rank-biserial effect sizes / AUC for qualitative citation. Not a model,
not a claim of mechanism — a check on whether "early-layer, context-carried
disambiguation" is a real, measurable regularity rather than a cherry-pick.

Usage:
  python -m helpers.analyze_wic_patterns --in_file outputs/wic/wic_dlig.jsonl
  python -m helpers.analyze_wic_patterns --in_file outputs/wic/wic_dlig.jsonl --csv outputs/wic/features.csv
"""

import os
import json
import argparse
import numpy as np


# ---- reuse the exact attribution reduction plot_wic uses -------------------

def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def signed_d_by_layer(row, step_filter=None):
    """{layer: mean-over-steps score vector}. Mirrors plot_wic.signed_d_by_layer."""
    acc, cnt = {}, {}
    for sd in row["steps_data"]:
        if step_filter is not None and sd["step"] != step_filter:
            continue
        for layer_str, scores in sd["layers"].items():
            l = int(layer_str)
            v = np.array(scores, dtype=float)
            if l not in acc:
                acc[l] = np.zeros_like(v)
                cnt[l] = 0
            acc[l] += v
            cnt[l] += 1
    return {l: acc[l] / max(cnt[l], 1) for l in acc}


def _clean_token(t):
    return t.replace("\u0120", "").replace("_", "").lstrip().lower()


# ---- token-region masks (positional, not semantic) ------------------------

def pivot_mask(tokens, word):
    """Tokens whose cleaned form exactly matches the pivot word.
    Exact-only: the earlier substring variant over-matched short words and
    inflated pivot_share; exact matching gave a cleaner, stronger effect."""
    w = word.strip().lower()
    m = np.zeros(len(tokens), dtype=bool)
    for i, t in enumerate(tokens):
        if _clean_token(t) == w:
            m[i] = True
    return m


def sentence_split(tokens):
    """
    Index that splits sentence-1 from sentence-2 context. Positional heuristic:
    the *second* sentence-terminal punctuation ('.', '?', '!') — WiC prompts hold
    two example sentences. Falls back to the midpoint. Returns cut index.
    """
    terms = [i for i, t in enumerate(tokens)
             if _clean_token(t) in {".", "?", "!"}]
    if len(terms) >= 2:
        return terms[len(terms) // 2 - 1] + 1  # after the ~first-half terminator
    return len(tokens) // 2


# ---- per-example features, all derived from |attr| geometry ----------------

def example_features(row):
    dbl = signed_d_by_layer(row)
    layers = sorted(dbl)
    if not layers:
        return None
    tokens = row["input_tokens"]
    n = len(tokens)

    # matrix: layers x tokens (signed); and its abs
    D = np.vstack([dbl[l] for l in layers])          # (L, T)
    A = np.abs(D)
    total = A.sum() + 1e-12

    # 1. layer centroid of mass (0..1 over the sampled layer axis)
    per_layer = A.sum(axis=1)                          # (L,)
    lay_norm = np.array(layers, dtype=float)
    lay_norm = (lay_norm - lay_norm.min()) / (lay_norm.max() - lay_norm.min() + 1e-12)
    layer_centroid = float((per_layer * lay_norm).sum() / (per_layer.sum() + 1e-12))

    # 2. mass fractions in shallow vs deep layer bands
    shallow = [i for i, l in enumerate(layers) if l <= 8]
    deep    = [i for i, l in enumerate(layers) if l >= 20]
    frac_shallow = float(per_layer[shallow].sum() / (per_layer.sum() + 1e-12)) if shallow else np.nan
    frac_deep    = float(per_layer[deep].sum()    / (per_layer.sum() + 1e-12)) if deep    else np.nan

    # 3. pivot share of mass (pivot token identity from row["word"], not hand rule)
    pm = pivot_mask(tokens, row.get("word", ""))
    pivot_share = float(A[:, pm].sum() / total) if pm.any() else 0.0

    # 4. context-vs-pivot ratio (how much lives off the pivot)
    ctx_share = 1.0 - pivot_share
    ctx_pivot_ratio = float(ctx_share / (pivot_share + 1e-3))

    # 5. sign concentration: entropy of the |attr| distribution over tokens
    #    (low entropy = peaked on a few tokens; high = diffuse)
    p = A.sum(axis=0)
    p = p / (p.sum() + 1e-12)
    tok_entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(n))  # 0..1

    # 6. sentence-2 vs sentence-1 mass balance (positional split)
    cut = sentence_split(tokens)
    s1 = A[:, :cut].sum()
    s2 = A[:, cut:].sum()
    sent2_frac = float(s2 / (s1 + s2 + 1e-12))

    # 7. net signed polarity (does mass lean toward or against committed answer)
    signed_mean = float(D.sum() / total)

    return dict(
        idx=row["idx"], word=row.get("word", ""),
        label=int(row["label"]),
        pred=(int(row["pred"]) if row["pred"] is not None else None),
        correct=bool(row["correct"]),
        layer_centroid=layer_centroid,
        frac_shallow=frac_shallow, frac_deep=frac_deep,
        pivot_share=pivot_share, ctx_pivot_ratio=ctx_pivot_ratio,
        tok_entropy=tok_entropy, sent2_frac=sent2_frac,
        signed_polarity=signed_mean,
    )


# ---- stats: AUC + Mann-Whitney, no sklearn dependency ----------------------

def auc_mw(pos, neg):
    """AUC (= P(pos>neg)) and Mann-Whitney U p-value via normal approx."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, (n1, n2)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    tie_rank = np.zeros(len(_))
    cum = 0
    for k, c in enumerate(cnt):
        tie_rank[k] = cum + (c + 1) / 2.0
        cum += c
    ranks = tie_rank[inv]
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    auc = U1 / (n1 * n2)
    mu = n1 * n2 / 2.0
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (U1 - mu) / (sd + 1e-12)
    # two-sided p via erfc
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2))
    return float(auc), float(p), (n1, n2)


FEATURES = ["layer_centroid", "frac_shallow", "frac_deep", "pivot_share",
            "ctx_pivot_ratio", "tok_entropy", "sent2_frac", "signed_polarity"]


def contrast(feats, name, pos_pred, neg_pred):
    pos = [f for f in feats if pos_pred(f)]
    neg = [f for f in feats if neg_pred(f)]
    print(f"\n=== {name}  (pos={len(pos)}, neg={len(neg)}) ===")
    if not pos or not neg:
        print("  [skipped: empty group]")
        return
    print(f"{'feature':<18}{'AUC':>8}{'|AUC-.5|':>10}{'p':>10}")
    rows = []
    for k in FEATURES:
        auc, p, _ = auc_mw([f[k] for f in pos], [f[k] for f in neg])
        rows.append((abs(auc - 0.5) if not np.isnan(auc) else -1, k, auc, p))
    for eff, k, auc, p in sorted(rows, reverse=True):
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""
        print(f"{k:<18}{auc:>8.3f}{abs(auc-0.5):>10.3f}{p:>10.4f}  {star}")


def confusion(feats):
    from collections import Counter
    c = Counter((f["label"], f["pred"]) for f in feats)
    print("\n[confusion] (label, pred) counts:")
    for k in sorted(c, key=lambda x: (x[0], str(x[1]))):
        print(f"    label={k[0]} pred={k[1]}: {c[k]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default="outputs/wic/wic_dlig.jsonl")
    ap.add_argument("--csv", default=None, help="optional: dump per-example features")
    args = ap.parse_args()

    rows = load_rows(args.in_file)
    feats = [f for f in (example_features(r) for r in rows) if f is not None]
    # pred can be None (model emitted neither Yes nor No); tag those out of the
    # correct/incorrect contrasts so they don't masquerade as a predicted class.
    for f in feats:
        f["parsed"] = f["pred"] in (0, 1)
    n_none = sum(not f["parsed"] for f in feats)
    print(f"[info] {len(feats)} examples with attribution "
          f"({n_none} with unparseable pred, excluded from pred-based contrasts).")

    confusion(feats)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w") as fh:
            cols = [c for c in feats[0].keys()]
            fh.write(",".join(cols) + "\n")
            for f in feats:
                fh.write(",".join(str(f[c]) for c in cols) + "\n")
        print(f"[info] wrote {args.csv}")

    P = [f for f in feats if f["parsed"]]  # parseable only for pred-based splits

    # --- NO class: read-the-context (correct) vs Yes-bias default (incorrect) --
    contrast(P, "gold-No: correct=said-No (pos) vs incorrect=Yes-bias (neg)",
             lambda f: f["label"] == 0 and f["correct"],
             lambda f: f["label"] == 0 and not f["correct"])

    # --- YES class: correct vs the rare Yes-error --------------------------------
    contrast(P, "gold-Yes: correct=said-Yes (pos) vs incorrect=said-No (neg)",
             lambda f: f["label"] == 1 and f["correct"],
             lambda f: f["label"] == 1 and not f["correct"])

    # --- WHEN THE MODEL SAYS YES: real Yes vs defaulted Yes ----------------------
    # The bias test: among predicted-Yes, does attribution distinguish a genuine
    # same-sense read (gold-Yes) from a defaulted Yes on a different-sense item?
    contrast(P, "predicted-Yes: genuine=gold-Yes (pos) vs defaulted=gold-No (neg)",
             lambda f: f["pred"] == 1 and f["label"] == 1,
             lambda f: f["pred"] == 1 and f["label"] == 0)

    # --- WHEN THE MODEL SAYS NO: real No vs the rare mistaken No -----------------
    contrast(P, "predicted-No: genuine=gold-No (pos) vs mistaken=gold-Yes (neg)",
             lambda f: f["pred"] == 0 and f["label"] == 0,
             lambda f: f["pred"] == 0 and f["label"] == 1)

    # --- pooled: does attribution geometry track correctness at all -------------
    contrast(P, "pooled: correct (pos) vs incorrect (neg)",
             lambda f: f["correct"], lambda f: not f["correct"])

    print("\nAUC ~0.5 = no separation; further from .5 = feature tracks the split.")
    print("Read effect sizes qualitatively; p is descriptive (8 features x 5 "
          "contrasts, no multiple-comparison correction).")


if __name__ == "__main__":
    main()