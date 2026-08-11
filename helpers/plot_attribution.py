# utils/plot_absolute_attribution.py
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# Paper-figure defaults: figures render at single-column width (~3.3in), so
# fonts must be large relative to the canvas. No in-figure titles -- captions
# live in the LaTeX Figure environment; stats (r, n) are printed to stdout
# for the caption text.
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.titlesize": 16,
})


# layer aggregation: collapse the per-layer DLIG into one score per position
#   sum  : total attribution mass routed through that position (all depth)
#   mean : average over layers
#   peak : max over layers (dominant-layer readout)
def _collapse_layers(layers_dict, n_pos, mode="sum"):
    """layers_dict: {layer_str: [score per position]} -> np.array[n_pos]."""
    acc = np.zeros(n_pos, dtype=np.float64)
    if mode == "peak":
        acc[:] = -np.inf
    count = 0
    for _layer_str, token_scores in layers_dict.items():
        ts = np.asarray(token_scores[:n_pos], dtype=np.float64)
        if ts.shape[0] < n_pos:
            ts = np.pad(ts, (0, n_pos - ts.shape[0]))
        if mode == "sum" or mode == "mean":
            acc += np.abs(ts)
        elif mode == "peak":
            acc = np.maximum(acc, np.abs(ts))
        count += 1
    if mode == "mean" and count > 0:
        acc /= count
    if mode == "peak":
        acc[~np.isfinite(acc)] = 0.0
    return acc


def aggregate_infill(input_file, layer_agg="sum", split_rouge=None,
                     normalize=True, tail_min=0, mass_per_token=False):
    """ROCStories infilling schema. Per row: signed_dist, rouge1,
    steps_data:[{step, layers:{l:[score per kept ctx pos]}}].

    Returns (by_step, meta) where:
      by_step = {step: {group_key: {signed_dist: [scores]}}}
      meta    = {
        'counts': {group: n_stories},
        'per_story': [ {step, rouge1, left_mass, right_mass, ratio} ... ],  # per (story,step)
      }

    normalize: per-story max-normalize each step's position scores (shape only).
               Set False to keep raw meancentered magnitudes (group magnitude
               differences survive — needed to see if good infills carry MORE
               right-context mass, not just a different shape).
    tail_min:  if >0, only positions with |signed_dist| >= tail_min contribute
               (excludes the boundary spike so the tail asymmetry is visible).
    """
    by_step = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    counts = defaultdict(set)
    per_story = []

    if not os.path.exists(input_file):
        print(f"[ERROR] missing: {input_file}")
        return by_step, {"counts": {}, "per_story": []}

    with open(input_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sdist = row.get("signed_dist", [])
            n = len(sdist)
            if n == 0:
                continue
            r1 = row.get("rouge1", 0.0)
            sid = row.get("story_id", "")
            if split_rouge is not None:
                key = "high" if r1 >= split_rouge else "low"
            else:
                key = "all"
            counts[key].add(sid)

            sdist_arr = np.asarray(sdist)
            for sd in row.get("steps_data", []):
                step = sd.get("step", 0)
                layers_dict = sd.get("layers", {})
                if not layers_dict:
                    continue
                # self-mode degenerate cells: no scoreable committed span token
                # => F == 0 => zero attribution. New rows carry skipped=True /
                # n_scoreable=0; old-format rows carry all-zero scores. Skip all.
                if sd.get("skipped", False) or sd.get("n_scoreable", None) == 0:
                    continue
                n_scoreable = sd.get("n_scoreable", None)
                pos = _collapse_layers(layers_dict, n, mode=layer_agg)
                pos = np.abs(pos)
                if np.max(pos) == 0.0:
                    continue                    # all-zero cell (old-format degenerate)
                if normalize:
                    pos = pos / np.max(pos)

                # per-story scalar: left vs right context mass (tail only, excludes
                # the boundary spike so it measures genuine context use)
                tmask = np.abs(sdist_arr) >= max(tail_min, 1)
                left_mask = tmask & (sdist_arr < 0)
                right_mask = tmask & (sdist_arr > 0)
                left_mass = float(pos[left_mask].sum())
                right_mass = float(pos[right_mask].sum())
                total_mass = left_mass + right_mass
                if mass_per_token and n_scoreable:
                    # self mode: the target token count grows over denoising
                    # steps; per-scoreable-token mass removes that trend.
                    left_mass /= n_scoreable
                    right_mass /= n_scoreable
                    total_mass /= n_scoreable
                ratio = right_mass / (left_mass + 1e-9)
                per_story.append({"step": step, "rouge1": r1,
                                  "left_mass": left_mass, "right_mass": right_mass,
                                  "total_mass": total_mass,
                                  "n_scoreable": n_scoreable,
                                  "ratio": ratio})

                for i in range(n):
                    if tail_min > 0 and abs(int(sdist[i])) < tail_min:
                        continue
                    by_step[step][key][int(sdist[i])].append(float(pos[i]))

    meta = {"counts": {k: len(v) for k, v in counts.items()}, "per_story": per_story}
    return by_step, meta


def _draw_infill_axis(ax, groups, cmap, max_abs_dist, bin_width, min_count):
    style = {
        "all": dict(color=cmap(0.75), marker='o', label="all"),
        "high": dict(color=cmap(0.85), marker='o', label="high ROUGE"),
        "low": dict(color="gray", marker='x', linestyle='--', label="low ROUGE"),
    }
    for key, agg in groups.items():
        dists = sorted(agg.keys())
        if not dists:
            continue
        md = max_abs_dist if max_abs_dist is not None else max(abs(min(dists)), abs(max(dists)))
        centers, means, stderrs = [], [], []
        edge = -((md // bin_width) * bin_width)
        while edge <= md:
            vals = []
            for d in range(edge, edge + bin_width):
                vals.extend(agg.get(d, []))
            if len(vals) >= min_count:
                centers.append(edge + (bin_width - 1) / 2.0)
                means.append(np.mean(vals))
                stderrs.append(np.std(vals) / np.sqrt(len(vals)))
            edge += bin_width
        st = style.get(key, dict(color=cmap(0.6), marker='o', label=key))
        ax.errorbar(centers, means, yerr=stderrs, markersize=3, linewidth=1.8,
                    capsize=2, alpha=0.9, **st)
    ax.axvline(0, color='black', linewidth=1, linestyle=':')
    ax.grid(axis='y', alpha=0.3)


def plot_infill(by_step, output_file, cmap_name="Purples",
                max_abs_dist=None, bin_width=2, min_count=5, panel_steps=None):
    """Small-multiples grid: one panel per denoising step, signed distance-from-
    span on x (x<0 left context s1,s2; x>0 right context s4,s5). No figure
    title: the caption lives in the paper. panel_steps: subset of steps to
    render (e.g. [1,5,11] for a single-row main-text figure); None = all."""
    if not by_step:
        print("[ABORT] no data.")
        return

    steps = sorted(by_step.keys())
    if panel_steps is not None:
        steps = [s for s in steps if s in set(panel_steps)]
    n = len(steps)
    if n == 0:
        print("[ABORT] no matching panel steps.")
        return
    cols = min(3, n) if n <= 3 else 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.6 * rows),
                             squeeze=False, sharey=True)
    axes = axes.flatten()
    cmap = plt.get_cmap(cmap_name)

    for i, step in enumerate(steps):
        ax = axes[i]
        _draw_infill_axis(ax, by_step[step], cmap, max_abs_dist, bin_width, min_count)
        ax.set_title(f"$t={step}$")
        if i // cols == rows - 1:
            ax.set_xlabel("Signed distance from span (tokens)")
        if i % cols == 0:
            ax.set_ylabel("Normalized |DLIG|")
        if i == 0:
            ax.legend(frameon=False, loc="upper right")
            ax.text(0.02, 0.03, r"$\leftarrow$ left", transform=ax.transAxes,
                    fontsize=12, va='bottom')
            ax.text(0.80, 0.03, r"right $\rightarrow$", transform=ax.transAxes,
                    fontsize=12, va='bottom')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] infill trajectory profile ({n} panels) -> {output_file}")


def plot_scalar_scatter(per_story, field, ylabel, output_file,
                        step=None, href=None, logy=False):
    """Per-story <field> vs ROUGE-1 scatter with binned trend. No in-figure
    title; Pearson r and n are printed to stdout for the LaTeX caption.
    href: optional horizontal reference line. step: restrict to one denoising step."""
    rows = [d for d in per_story if (step is None or d["step"] == step)]
    if not rows:
        print("[ABORT] no per-story data for scatter.")
        return float("nan")
    x = np.array([d["rouge1"] for d in rows])
    y = np.array([d[field] for d in rows], dtype=float)
    y_disp = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))

    r = np.corrcoef(x, y)[0, 1] if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(x, y_disp, s=10, alpha=0.4, color="#6b3fa0")
    if len(x) > 10:
        order = np.argsort(x)
        xs, ys = x[order], y_disp[order]
        nb = min(10, len(xs) // 5)
        if nb >= 2:
            edges = np.linspace(xs.min(), xs.max(), nb + 1)
            bx, by_ = [], []
            for k in range(nb):
                m = (xs >= edges[k]) & (xs <= edges[k + 1])
                if m.sum() >= 3:
                    bx.append(xs[m].mean()); by_.append(ys[m].mean())
            ax.plot(bx, by_, color="black", linewidth=2, marker="o", label="binned mean")
            ax.legend(frameon=False)
    if href is not None:
        ax.axhline(href, color="gray", linestyle=":", linewidth=1)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("ROUGE-1 (infill quality)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    step_tag = f"step={step}" if step is not None else "all steps"
    print(f"[SUCCESS] {field}-vs-ROUGE scatter -> {output_file}")
    print(f"[CAPTION] {field}: Pearson r={r:.3f}, n={len(x)}, {step_tag}")
    return r


def plot_r_vs_step(per_story, field, output_file):
    """Pearson r(<field>, ROUGE-1) computed PER denoising step, plotted vs step.
    No in-figure title; per-step r and n printed to stdout for the caption.
    Per-step n varies under the self-generated target (degenerate early cells
    are dropped), so n is reported alongside r."""
    steps = sorted({d["step"] for d in per_story})
    rs, ns = [], []
    for s in steps:
        rows = [d for d in per_story if d["step"] == s]
        x = np.array([d["rouge1"] for d in rows])
        y = np.array([d[field] for d in rows], dtype=float)
        r = np.corrcoef(x, y)[0, 1] if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")
        rs.append(r); ns.append(len(rows))
    # approx 95% CI for Pearson r: 1.96 / sqrt(n-3)
    ci = [1.96 / np.sqrt(max(n - 3, 1)) for n in ns]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.errorbar(steps, rs, yerr=ci, marker="o", markersize=6, linewidth=2,
                capsize=4, color="#6b3fa0")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Denoising step $t$")
    ax.set_ylabel("Pearson $r$(mass, ROUGE-1)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    rtxt = ", ".join(f"t{s}: r={r:.3f} (n={n})" for s, r, n in zip(steps, rs, ns))
    print(f"[SUCCESS] r-vs-step -> {output_file}")
    print(f"[CAPTION] {field} per step: {rtxt}")
    return dict(zip(steps, rs))


def plot_ratio_scatter(per_story, output_file, step=None):
    """Appendix-only: right/left ratio vs ROUGE (null result; main text cites
    the r value instead of showing the figure)."""
    return plot_scalar_scatter(per_story, "ratio",
                               "Right / Left context-mass ratio",
                               output_file, step=step, href=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default="diffugpt", choices=["dream", "diffugpt"])
    parser.add_argument("--target_mode", type=str, default="self",
                        choices=["self", "gold"],
                        help="Which attribution run to plot: 'self' (primary, "
                             "self-generated targets) or 'gold' (fixed-target, "
                             "appendix robustness). Selects the default input file "
                             "and tags the output plot names.")
    parser.add_argument("--panel_steps", type=int, nargs="+", default=None,
                        help="infill profile: subset of denoising steps to render "
                             "(e.g. 1 5 11 for a single-row main-text figure). "
                             "Default: all recorded steps.")
    parser.add_argument("--mass_per_token", action="store_true",
                        help="normalize left/right/total context mass by the number "
                             "of scoreable target tokens at each step (self mode: "
                             "removes the growing-target-size trend across steps).")
    parser.add_argument("--out_dir", type=str, default="outputs/absolute")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--layer_agg", type=str, default="sum",
                        choices=["sum", "mean", "peak"],
                        help="How to collapse per-layer DLIG into one score per position.")
    parser.add_argument("--min_count", type=int, default=5,
                        help="infill: drop distance bins with fewer samples.")
    parser.add_argument("--split_rouge", type=float, default=None,
                        help="infill: ROUGE-1 threshold to split high/low-quality infills.")
    parser.add_argument("--raw", action="store_true",
                        help="infill: do NOT per-story normalize (keep meancentered magnitudes "
                             "so group magnitude differences survive).")
    parser.add_argument("--tail_min", type=int, default=0,
                        help="infill: only |signed_dist| >= tail_min (excludes the boundary "
                             "spike so tail/context asymmetry is visible).")
    parser.add_argument("--max_dist", type=int, default=None,
                        help="Cap on distance-from-span axis.")
    parser.add_argument("--bin_width", type=int, default=2,
                        help="Bin width over distance (smooths variable lengths).")
    parser.add_argument("--scatter", action="store_true",
                        help="infill: also emit right/left mass-ratio vs ROUGE-1 scatter (the "
                             "sensitive quality test, with Pearson r).")
    parser.add_argument("--mass_scatter", action="store_true",
                        help="infill: emit TOTAL context-mass vs ROUGE-1 scatter + the "
                             "r(total_mass,ROUGE)-vs-denoising-step curve (the magnitude/trajectory test).")
    parser.add_argument("--scatter_step", type=int, default=None,
                        help="infill scatter: restrict single-scatter to one denoising step (default: all).")
    args = parser.parse_args()

    fam_display = "Dream-7B" if args.family.lower() == "dream" else "DiffuGPT-M"
    input_file = args.input_file or os.path.join(
        args.out_dir,
        f"{args.family.lower()}_rocstories_infill_attribution_{args.target_mode}.jsonl")

    norm_tag = "_raw" if args.raw else ""
    tail_tag = f"_tail{args.tail_min}" if args.tail_min else ""
    tag = f"_split{args.split_rouge}" if args.split_rouge is not None else ""
    mode_tag = f"_{args.target_mode}"
    panel_tag = ("_t" + "-".join(map(str, args.panel_steps))) if args.panel_steps else ""
    output_plot = os.path.join(
        args.out_dir,
        f"plot_{args.family.lower()}_rocstories_infill{mode_tag}{tag}{norm_tag}{tail_tag}{panel_tag}.png")
    print(f"[INFO] {fam_display} ROCStories infilling from {input_file} "
          f"[mode={args.target_mode}, split_rouge={args.split_rouge}, raw={args.raw}, "
          f"tail_min={args.tail_min}]...")
    by_step, meta = aggregate_infill(
        input_file, layer_agg=args.layer_agg, split_rouge=args.split_rouge,
        normalize=(not args.raw), tail_min=args.tail_min,
        mass_per_token=args.mass_per_token)

    # ---- group sizes (diagnose a degenerate split) ----
    print(f"[COUNTS] stories per group: {meta['counts']}")
    if args.split_rouge is not None:
        ps = meta["per_story"]
        if ps:
            r1s = np.array([d["rouge1"] for d in ps])
            print(f"[ROUGE]  mean={r1s.mean():.3f} median={np.median(r1s):.3f} "
                  f"min={r1s.min():.3f} max={r1s.max():.3f} "
                  f">={args.split_rouge}: {(r1s>=args.split_rouge).mean()*100:.1f}%")

    plot_infill(
        by_step,
        output_plot,
        cmap_name="Purples" if args.family.lower() == "diffugpt" else "Teals",
        max_abs_dist=args.max_dist, bin_width=args.bin_width,
        min_count=args.min_count, panel_steps=args.panel_steps,
    )

    if args.scatter:
        scat_out = os.path.join(
            args.out_dir,
            f"plot_{args.family.lower()}_rocstories_infill{mode_tag}_ratio_scatter"
            f"{('_step'+str(args.scatter_step)) if args.scatter_step is not None else ''}.png")
        plot_ratio_scatter(meta["per_story"], scat_out, step=args.scatter_step)

    if args.mass_scatter:
        # total-context-mass vs ROUGE scatter (single step or pooled)
        ms_out = os.path.join(
            args.out_dir,
            f"plot_{args.family.lower()}_rocstories_infill{mode_tag}_totalmass_scatter"
            f"{('_step'+str(args.scatter_step)) if args.scatter_step is not None else ''}.png")
        plot_scalar_scatter(
            meta["per_story"], "total_mass",
            "Total context mass (|DLIG| over context)",
            ms_out, step=args.scatter_step)
        # r(total_mass, ROUGE) vs denoising step  -- the trajectory result
        rstep_out = os.path.join(
            args.out_dir,
            f"plot_{args.family.lower()}_rocstories_infill{mode_tag}_r_vs_step.png")
        plot_r_vs_step(meta["per_story"], "total_mass", rstep_out)


if __name__ == "__main__":
    main()