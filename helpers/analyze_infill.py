# helpers/analyze_infill.py
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

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
                pos_raw = np.abs(_collapse_layers(layers_dict, n, mode=layer_agg))
                if np.max(pos_raw) == 0.0:
                    continue                    # all-zero cell (old-format degenerate)
                # Profile curves use shape-normalized scores when requested;
                # per-story masses are ALWAYS computed on raw magnitudes so the
                # mass-vs-quality analyses are magnitude analyses regardless of
                # the profile normalization flag.
                pos = pos_raw / np.max(pos_raw) if normalize else pos_raw

                # per-story scalar: left vs right context mass (tail only, excludes
                # the boundary spike so it measures genuine context use)
                tmask = np.abs(sdist_arr) >= max(tail_min, 1)
                left_mask = tmask & (sdist_arr < 0)
                right_mask = tmask & (sdist_arr > 0)
                left_mass = float(pos_raw[left_mask].sum())
                right_mass = float(pos_raw[right_mask].sum())
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


def report_empty_steps(input_file, eps=1e-9):
    """Per-step surviving-n (n_scoreable>0) vs dropped (F_t==0) for the
    self-generated framing — the cells aggregate_infill() skips via
    `if sd.get("skipped") or sd.get("n_scoreable") == 0: continue`. Regenerates
    the §7 footnote sample sizes (n=544 at t=1, n>=981 at t>=5) from the jsonl.

    Also cross-tabs n_scoreable==0 against total-mass<=eps: on this data the two
    coincide exactly (no off-diagonal cells), which is what lets §7 exclude
    degenerate cells 'for the same reason as' the WiC depth footnote (§5.1) — the
    two scripts detect the no-committed-target condition by different signals
    (token count vs. total mass) that happen to select the identical cells."""
    if not os.path.exists(input_file):
        print(f"[ERROR] missing: {input_file}")
        return None
    surv, dropp = {}, {}
    ct = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    with open(input_file) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if len(row.get("signed_dist", [])) == 0:
                continue
            for sd in row.get("steps_data", []):
                s = sd["step"]
                nsc = sd.get("n_scoreable", None)
                lay = sd.get("layers", {})
                m = (sum(np.abs(np.array(v, float)).sum() for v in lay.values())
                     if lay else 0.0)
                sc0 = bool(sd.get("skipped", False) or nsc == 0)
                m0 = (m <= eps)
                ct[(sc0, m0)] += 1
                d = dropp if sc0 else surv
                d[s] = d.get(s, 0) + 1
    steps = sorted(set(surv) | set(dropp))
    print(f"[infill empty-step report] eps={eps:g}")
    print(f"  {'step':>4} {'surv':>6} {'dropped':>7} {'total':>6}")
    for s in steps:
        sv, dp = surv.get(s, 0), dropp.get(s, 0)
        print(f"  {s:>4} {sv:>6} {dp:>7} {sv+dp:>6}")
    print(f"  footnote: surviving n per step -> "
          f"{{{', '.join(f't{s}:{surv.get(s, 0)}' for s in steps)}}}")
    print(f"  cross-tab (n_scoreable==0 vs mass<=eps): "
          f"both0={ct[(True, True)]}, sc0&mass>0={ct[(True, False)]}, "
          f"sc>0&mass0={ct[(False, True)]}, both>0={ct[(False, False)]}")
    agree = ct[(True, False)] == 0 and ct[(False, True)] == 0
    print(f"  -> criteria {'COINCIDE exactly' if agree else 'DIVERGE'} on this data")
    return {"surviving": surv, "dropped": dropp, "crosstab": ct}


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
    # Prefer a wide 3-across grid (6 panels -> 3 cols x 2 rows). Panels are
    # sized close to square and share both axes so interior tick labels are
    # dropped, which removes most of the inter-panel whitespace.
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.7 * rows),
                             squeeze=False, sharey=True, sharex=True)
    axes = axes.flatten()
    cmap = plt.get_cmap(cmap_name)

    for i, step in enumerate(steps):
        ax = axes[i]
        _draw_infill_axis(ax, by_step[step], cmap, max_abs_dist, bin_width, min_count)
        ax.set_title(f"$t={step}$", pad=3)
        if i == 0:
            # Anchor the center of the legend to data coordinates (x=0, y=1)
            ax.legend(frameon=False, loc="center", bbox_to_anchor=(0, 1), bbox_transform=ax.transData)
            ax.text(0.02, 0.03, r"$\leftarrow$ left", transform=ax.transAxes,
                    fontsize=12, va='bottom')
            ax.text(0.80, 0.03, r"right $\rightarrow$", transform=ax.transAxes,
                    fontsize=12, va='bottom')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Single shared axis labels (sharex/sharey make per-panel labels redundant
    # and, repeated across 3 columns, they collide). One centered label each.
    fig.supxlabel("Signed distance from span (tokens)")
    fig.supylabel("Normalized |DLIG|")

    # tight_layout with small pads, then squeeze the sharey/sharex gaps to near-0
    plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.5)
    plt.subplots_adjust(wspace=0.05, hspace=0.18)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] infill trajectory profile ({n} panels, "
          f"{rows}x{cols}) -> {output_file}")


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
    No in-figure title; per-step r, 95% bootstrap CI, and n printed to stdout
    for the caption. Per-step n varies under the self-generated target
    (degenerate early cells are dropped), so error bars are a percentile
    bootstrap over stories rather than the null 1.96/sqrt(n-3) width -- this
    keeps the early-step CI honest about the reduced, differently-composed
    sample."""
    steps = sorted({d["step"] for d in per_story})
    if not steps:
        print("[ABORT] no per-story data for r-vs-step.")
        return {}
    rng = np.random.default_rng(0)
    n_boot = 2000
    rs, ns, lo, hi = [], [], [], []
    for s in steps:
        rows = [d for d in per_story if d["step"] == s]
        x = np.array([d["rouge1"] for d in rows])
        y = np.array([d[field] for d in rows], dtype=float)
        ok = len(x) > 2 and x.std() > 0 and y.std() > 0
        r = np.corrcoef(x, y)[0, 1] if ok else float("nan")
        rs.append(r); ns.append(len(rows))
        # Percentile bootstrap CI over stories: honest about the per-step sample
        # (early self-mode steps drop degenerate cells, so n and composition vary).
        if ok:
            idx = np.arange(len(x))
            boot = np.empty(n_boot)
            for b in range(n_boot):
                bi = rng.choice(idx, size=len(idx), replace=True)
                bx, by = x[bi], y[bi]
                boot[b] = (np.corrcoef(bx, by)[0, 1]
                           if bx.std() > 0 and by.std() > 0 else np.nan)
            lo.append(np.nanpercentile(boot, 2.5))
            hi.append(np.nanpercentile(boot, 97.5))
        else:
            lo.append(float("nan")); hi.append(float("nan"))
    # asymmetric yerr for errorbar: distances from the point estimate
    yerr = np.array([[r - l for r, l in zip(rs, lo)],
                     [h - r for r, h in zip(rs, hi)]])

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.errorbar(steps, rs, yerr=yerr, marker="o", markersize=6, linewidth=2,
                capsize=4, color="#6b3fa0")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Denoising step $t$")
    ax.set_ylabel("Pearson $r$(mass, ROUGE-1)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    rtxt = ", ".join(f"t{s}: r={r:.3f} [{l:.3f},{h:.3f}] (n={n})"
                     for s, r, l, h, n in zip(steps, rs, lo, hi, ns))
    print(f"[SUCCESS] r-vs-step -> {output_file}")
    print(f"[CAPTION] {field} per step (2000-boot 95% CI): {rtxt}")
    return dict(zip(steps, rs))


def _mannwhitney_auc_p(a, b, n_boot=2000, seed=0):
    """AUC that a random high-ROUGE story (a) outranks a random low-ROUGE story
    (b) on the given mass, with a two-sided normal-approx p and a percentile
    bootstrap 95% CI. 0.5 = no difference; >0.5 = high-ROUGE carries more mass.
    Pure-numpy (no scipy dependency), tie-corrected rank-sum."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), n1, n2
    concat = np.concatenate([a, b])
    ranks = _rankdata_avg(concat)                    # average ranks (tie-safe)
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    auc = u1 / (n1 * n2)
    # two-sided normal approx with tie correction on the U statistic
    _, counts = np.unique(concat, return_counts=True)
    n = n1 + n2
    tie = (counts ** 3 - counts).sum()
    sd = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))) if n > 1 else 0.0
    mu = n1 * n2 / 2.0
    z = (u1 - mu) / sd if sd > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    # percentile bootstrap CI over stories (resample each group independently)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        ba = a[rng.integers(0, n1, n1)]
        bb = b[rng.integers(0, n2, n2)]
        rr = _rankdata_avg(np.concatenate([ba, bb]))
        boot[k] = (rr[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(auc), float(p), float(lo), float(hi), n1, n2


def _rankdata_avg(x):
    """Average ranks (1-based), ties share the mean of their rank positions."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def compute_mass_auc(per_story, split_rouge, field="total_mass"):
    """Two-group test: do high-ROUGE stories carry MORE context mass than
    low-ROUGE ones? Splits stories at `split_rouge` and reports Mann-Whitney
    AUC (high vs low) pooled over steps AND per denoising step, each with a
    95% bootstrap CI and two-sided p. Prints [CAPTION] lines for the appendix;
    returns a dict. This is the honest test behind the 'better infills show
    higher context dependence' claim -- the per-step AUC says whether the
    high>=low gap holds at 'almost every step' or only late in denoising."""
    if split_rouge is None:
        print("[WARN] compute_mass_auc needs --split_rouge; skipping.")
        return {}
    rows = [d for d in per_story if np.isfinite(d.get(field, np.nan))]
    if len(rows) < 6:
        print("[INFO] mass AUC: insufficient per-story data.")
        return {}

    def _split(rws):
        hi = [d[field] for d in rws if d["rouge1"] >= split_rouge]
        lo = [d[field] for d in rws if d["rouge1"] < split_rouge]
        return hi, lo

    out = {}
    hi, lo = _split(rows)
    auc, p, clo, chi, nh, nl = _mannwhitney_auc_p(hi, lo)
    out["pooled"] = (auc, p, clo, chi, nh, nl)
    # NOTE: the pooled test stacks all (story, step) rows, so one story's steps
    # are counted as independent -> n is inflated and the pooled p is optimistic.
    # Prefer the per-step AUCs below (each story contributes once per step) when
    # wording the 'at almost every step' claim.
    print(f"[CAPTION] {field} high-vs-low ROUGE AUC (pooled; pseudo-replicated, "
          f"see per-step): AUC={auc:.3f} [{clo:.3f},{chi:.3f}]  p={p:.2e}  "
          f"(n_high={nh}, n_low={nl})  [0.5=no diff, >0.5=high carries more]")

    steps = sorted({d["step"] for d in rows})
    print(f"[CAPTION] {field} high-vs-low ROUGE AUC per step:")
    for s in steps:
        hi_s, lo_s = _split([d for d in rows if d["step"] == s])
        a, pp, l, h, nhs, nls = _mannwhitney_auc_p(hi_s, lo_s, seed=s)
        out[s] = (a, pp, l, h, nhs, nls)
        star = "***" if pp < 1e-3 else "**" if pp < 1e-2 else "*" if pp < 5e-2 else "n.s."
        print(f"    t={s:2d}: AUC={a:.3f} [{l:.3f},{h:.3f}]  p={pp:.2e} {star:>4}  "
              f"(n_high={nhs}, n_low={nls})")
    return out


def compute_ratio_r(per_story, step=None):
    """Right/left mass-ratio vs ROUGE is a null result reported as a Pearson r
    in text, not as a figure. Compute and print r, n, and mean ratio; emit no
    plot. (Replaces the redundant ratio_scatter figure.)"""
    rows = [d for d in per_story if (step is None or d["step"] == step)]
    if len(rows) < 3:
        print("[INFO] ratio r: insufficient per-story data.")
        return float("nan")
    x = np.array([d["rouge1"] for d in rows])
    y = np.array([d["ratio"] for d in rows], dtype=float)
    r = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else float("nan")
    mean_ratio = float(np.mean(y))
    step_tag = f"step={step}" if step is not None else "all steps"
    print(f"[CAPTION] ratio(right/left) vs ROUGE: Pearson r={r:.3f}, "
          f"n={len(x)}, mean ratio={mean_ratio:.3f}, {step_tag}")
    return r


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
    parser.add_argument("--out_dir", type=str, default="outputs/infill_attribution")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--layer_agg", type=str, default="sum",
                        choices=["sum", "mean", "peak"],
                        help="How to collapse per-layer DLIG into one score per position.")
    parser.add_argument("--min_count", type=int, default=5,
                        help="infill: drop distance bins with fewer samples.")
    parser.add_argument("--split_rouge", type=str, default=None,
                        help="infill: ROUGE-1 threshold to split high/low-quality "
                             "infills. Pass a float, or 'median' to compute the "
                             "per-story median from the input file.")
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
    parser.add_argument("--ratio_r", action="store_true",
                        help="infill: print Pearson r(right/left ratio, ROUGE-1), n, and "
                             "mean ratio for the caption. Null result -- reported as a "
                             "number, no figure (replaces the redundant ratio_scatter).")
    parser.add_argument("--mass_auc", action="store_true",
                        help="infill: print Mann-Whitney AUC for total context mass, "
                             "high-vs-low ROUGE cohorts, pooled AND per denoising step "
                             "(with bootstrap CIs and p). Tests whether better infills "
                             "actually carry more mass; needs --split_rouge.")
    parser.add_argument("--mass_scatter", action="store_true",
                        help="infill: emit TOTAL context-mass vs ROUGE-1 scatter + the "
                             "r(total_mass,ROUGE)-vs-denoising-step curve (the magnitude/trajectory test).")
    parser.add_argument("--scatter_step", type=int, default=None,
                        help="infill scatter: restrict single-scatter to one denoising step (default: all).")
    parser.add_argument("--profile", action="store_true",
                        help="emit the signed-distance profile figure (small-multiples "
                             "grid). Off by default so scatter-only runs don't "
                             "regenerate it as a side effect.")
    parser.add_argument("--empty_report", action="store_true",
                        help="print per-step surviving/dropped cell counts (the "
                             "F_t==0 cells aggregate_infill skips) plus the "
                             "zero-scoreable/zero-mass cross-tab, then exit; "
                             "regenerates the §7 footnote sample sizes")

    args = parser.parse_args()

    fam_display = "Dream-7B" if args.family.lower() == "dream" else "DiffuGPT-M"
    input_file = args.input_file or os.path.join(
        args.out_dir,
        f"{args.family.lower()}_{args.target_mode}.jsonl")

    if args.empty_report:
        report_empty_steps(input_file)
        return

    # resolve --split_rouge: float, or 'median' computed over stories in the file
    if args.split_rouge is not None:
        if str(args.split_rouge).lower() == "median":
            r1s = []
            with open(input_file) as f:
                for line in f:
                    if line.strip():
                        r1s.append(json.loads(line).get("rouge1", 0.0))
            args.split_rouge = float(np.median(r1s))
            print(f"[INFO] split_rouge=median resolved to {args.split_rouge:.4f} "
                  f"over {len(r1s)} stories")
        else:
            args.split_rouge = float(args.split_rouge)

    norm_tag = "_raw" if args.raw else ""
    tail_tag = f"_tail{args.tail_min}" if args.tail_min else ""
    # Clean 2-decimal split tag: avoids float noise like split0.20000000000000004
    tag = f"_split{args.split_rouge:.2f}" if args.split_rouge is not None else ""
    mode_tag = f"_{args.target_mode}"
    mpt_tag = "_pertok" if args.mass_per_token else ""
    panel_tag = ("_t" + "-".join(map(str, args.panel_steps))) if args.panel_steps else ""
    plot_dir = os.path.join(args.out_dir, "plots", args.target_mode)
    os.makedirs(plot_dir, exist_ok=True)
    output_plot = os.path.join(
        plot_dir,
        f"{args.family.lower()}{mode_tag}{tag}{norm_tag}{tail_tag}{panel_tag}.png")
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

    if args.profile:
        plot_infill(
            by_step,
            output_plot,
            cmap_name="Purples" if args.family.lower() == "diffugpt" else "Teals",
            max_abs_dist=args.max_dist, bin_width=args.bin_width,
            min_count=args.min_count, panel_steps=args.panel_steps,
        )

    if args.ratio_r:
        compute_ratio_r(meta["per_story"], step=args.scatter_step)

    if args.mass_auc:
        compute_mass_auc(meta["per_story"], args.split_rouge, field="total_mass")

    if args.mass_scatter:
        # total-context-mass vs ROUGE scatter (single step or pooled)
        ms_out = os.path.join(
            plot_dir,
            f"{args.family.lower()}{mode_tag}{mpt_tag}_totalmass_scatter"
            f"{('_step'+str(args.scatter_step)) if args.scatter_step is not None else ''}.png")
        plot_scalar_scatter(
            meta["per_story"], "total_mass",
            "Total context mass",
            ms_out, step=args.scatter_step)
        # r(total_mass, ROUGE) vs denoising step  -- the trajectory result
        rstep_out = os.path.join(
            plot_dir,
            f"{args.family.lower()}{mode_tag}{mpt_tag}_r_vs_step.png")
        plot_r_vs_step(meta["per_story"], "total_mass", rstep_out)


if __name__ == "__main__":
    main()