#!/usr/bin/env python3
# helpers/plot_wic.py
"""
plot_wic.py — per-example, timestep-averaged, per-token DLIG bars paneled
across layers. The CMP-RT Figure-3 analog, for the WiC qualitative demonstration.

For a chosen example (by idx, or auto-picked from a population), draw one panel
per layer; within each panel, a signed bar per prompt token showing

    d[i] = DLIG(y+ = "### Yes")[i] - DLIG(y- = "### No")[i]

averaged over the target denoising steps. Positive (blue) = token supports
"same-sense"; negative (red) = supports "different-sense". This is the base-form
DLIG demonstration: no aggregation across examples, no token-role labels, no
statistical claim — just the attribution map the reader inspects.

Also supports an APPENDIX mode (--per_step): same bars but one row per denoising
step (not averaged), to visualize how attribution evolves over t with no claim.

Reads the JSONL from wic_contrastive_dlig.py.

Usage — main demo (pick 2-3 examples, timestep-averaged panels over layers):

python -m helpers.plot_wic --in_file outputs/wic/wic_dlig.jsonl --idx 208 --out_file outputs/wic/figs/correct_no_wall.png
python -m helpers.plot_wic --in_file outputs/wic/wic_dlig.jsonl --idx 264 --out_file outputs/wic/figs/correct_no_throw.png
python -m helpers.plot_wic --in_file outputs/wic/wic_dlig.jsonl --idx 116 --out_file outputs/wic/figs/correct_yes_love.png
python -m helpers.plot_wic --in_file outputs/wic/wic_dlig.jsonl --idx 168 --out_file outputs/wic/figs/correct_yes_channel.png

python -m helpers.plot_wic --in_file outputs/wic/wic_dlig.jsonl --idx 208 --per_step --out_file outputs/wic/figs/evolution_wall.png


  python -m helpers.plot_wic \
      --in_file outputs/wic/wic_dlig.jsonl \
      --idx 12 --out_file outputs/wic/figs/wic_idx12.png

  # or auto-pick a correct-No example (model overrode its Yes-bias):
  python -m helpers.plot_wic --in_file ... --pick correct_no

Appendix — per-step evolution for one example:
  python -m helpers.plot_wic --in_file ... --idx 12 --per_step \
      --out_file outputs/wic/figs/wic_idx12_steps.png
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pick_example(rows, idx, pick):
    if idx is not None:
        for r in rows:
            if r["idx"] == idx:
                return r
        raise SystemExit(f"idx {idx} not found in file.")
    # auto-pick by population
    def ok(r, want):
        if want == "correct_no":
            return r["correct"] and r["label"] == 0 and r["pred"] == 0
        if want == "correct_yes":
            return r["correct"] and r["label"] == 1 and r["pred"] == 1
        if want == "any_correct":
            return r["correct"]
        return True
    cands = [r for r in rows if ok(r, pick)]
    if not cands:
        raise SystemExit(f"no example matches --pick {pick}.")
    return cands[0]


def signed_d_by_layer(row, step_filter=None):
    """
    Return {layer: score_vector} where score = mean_over_steps(self-generated DLIG).
    Single target (the model's own answer) — NOT a plus/minus contrast.
    If step_filter is given, use just that step.
    """
    acc = {}
    cnt = {}
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
    """Strip GPT-2 BPE leading-space markers for display."""
    return t.replace("\u0120", "").replace("_", "").lstrip()


def _filter_tokens(tokens, d, min_frac):
    """
    Drop tokens whose |attribution| is a small fraction of this panel's max,
    so fat bars aren't crowded by near-zero noise. Returns (tokens, d) kept.
    Keeps at least the top-few so a near-empty panel doesn't vanish entirely.
    """
    d = np.asarray(d, dtype=float)
    m = np.max(np.abs(d)) if d.size else 0.0
    if m <= 0:
        return tokens, d
    thr = min_frac * m
    keep = np.abs(d) >= thr
    if keep.sum() < 3:  # guarantee a minimally readable panel
        keep = np.zeros_like(d, dtype=bool)
        keep[np.argsort(np.abs(d))[-3:]] = True
    kept_tokens = [tokens[i] for i in range(len(tokens)) if keep[i]]
    return kept_tokens, d[keep]


def _bar_panel(ax, tokens, d, title, min_frac):
    tokens, d = _filter_tokens(tokens, d, min_frac)
    tokens = [_clean_token(t) for t in tokens]
    x = np.arange(len(tokens))
    # single-target self-generated attribution: sign is the DLIG score sign
    # (positive = supports the committed answer, negative = against). Color by sign
    # for readability, but this is NOT a yes/no contrast.
    colors = ["#2c6fbb" if v >= 0 else "#c0392b" for v in d]
    ax.bar(x, d, color=colors, width=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.margins(x=0.01)
    # per-panel autoscale with a little headroom so small-attribution panels
    # (e.g. deep layers) remain informative instead of collapsing to a flat line.
    m = np.max(np.abs(d)) if d.size else 1.0
    ax.set_ylim(-1.15 * m, 1.15 * m)


def plot_layer_panels(row, layers, out_file, per_step=False, min_frac=0.05):
    tokens = row["input_tokens"]

    if not per_step:
        # MAIN: timestep-averaged, one panel per layer
        dbl = signed_d_by_layer(row)
        layers = [l for l in layers if l in dbl]
        n = len(layers)
        cols = min(4, n)
        rows_ = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_, cols, figsize=(4.2 * cols, 2.8 * rows_),
                                 squeeze=False)
        axes = axes.flatten()
        for i, l in enumerate(layers):
            _bar_panel(axes[i], tokens, dbl[l], f"layer {l}", min_frac)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
    else:
        # APPENDIX: per-step evolution. Rows = steps, cols = a few layers.
        steps = sorted({sd["step"] for sd in row["steps_data"]})
        show_layers = layers[:4] if len(layers) > 4 else layers
        fig, axes = plt.subplots(len(steps), len(show_layers),
                                 figsize=(4.2 * len(show_layers), 2.6 * len(steps)),
                                 squeeze=False)
        for si, st in enumerate(steps):
            dbl = signed_d_by_layer(row, step_filter=st)
            for li, l in enumerate(show_layers):
                if l in dbl:
                    _bar_panel(axes[si][li], tokens, dbl[l], f"t={st}, layer {l}", min_frac)
                else:
                    axes[si][li].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default="outputs/wic/wic_dlig.jsonl")
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--idx", type=int, default=None,
                    help="example idx to plot; if unset use --pick")
    ap.add_argument("--pick", default="correct_no",
                    choices=["correct_no", "correct_yes", "any_correct", "any"],
                    help="auto-pick population when --idx not given")
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[0, 4, 8, 12, 16, 20, 22],
                    help="which layers to panel (subset of what the runner stored)")
    ap.add_argument("--per_step", action="store_true",
                    help="appendix mode: show per-step evolution instead of averaging")
    ap.add_argument("--min_frac", type=float, default=0.05,
                    help="drop tokens with |attr| below this fraction of the panel max")
    args = ap.parse_args()

    rows = load_rows(args.in_file)
    row = pick_example(rows, args.idx, args.pick)
    plot_layer_panels(row, args.layers, args.out_file,
                      per_step=args.per_step, min_frac=args.min_frac)


if __name__ == "__main__":
    main()