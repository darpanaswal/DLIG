#!/usr/bin/env python3
# helpers/analyze_wic.py
"""
Unified WiC DLIG analysis. One script, two data sources, several modes.

Data sources (joined on idx):
  --dlig    wic_dlig.jsonl        -> attribution depth  (centroid)    [AXIS A]
  --commit  wic_commitment.jsonl  -> commitment timing  (commit_step) [AXIS B]

EXPERIMENT 1 (stats) — does the model's Yes response differ from its No response?
  Grouped by the model's PREDICTION (behaviour), not gold label.
  Reports AUC + Mann-Whitney for pred-Yes vs pred-No on each axis.

EXPERIMENT 2 (stats) — within the pred-Yes population, is there hidden structure?
  Unsupervised (no gold labels): Hartigan dip test + 1-vs-2 Gaussian BIC on each
  axis. Gold label is overlaid on the saved histogram as post-hoc colour only — a
  hypothesis for future work, never a claim.

Either axis runs alone if its file is absent. diptest is optional (pip install
diptest); without it Experiment 2 prints BIC only, which over-splits and must not
be trusted on its own.

FIGURE MODES (main-text figures for the paper):
  --depth_plot   attribution-depth centroid distribution (pred-Yes vs pred-No),
                 violin + box + jittered points. Reads --dlig. The "same depth"
                 result: the two response types read out from the same layers.
  --commit_plot  commitment trajectory: mean confidence in the committed answer
                 across denoising steps (pred-Yes vs pred-No), SEM bands, with the
                 0.5 commit threshold marked. Reads --commit. The "Yes commits
                 earlier" result.

PANEL MODE (--panel) — per-example, timestep-averaged, per-token DLIG bars
paneled across layers. The CMP-RT Figure-3 analog, for the WiC qualitative
demonstration. For a chosen example (by --panel_idx, or auto-picked via
--panel_pick), draws one panel per layer; within each panel, a signed bar per
prompt token showing

    d[i] = DLIG(y+ = "### Yes")[i] - DLIG(y- = "### No")[i]

averaged over the target denoising steps. Positive (blue) = token supports
"same-sense"; negative (red) = supports "different-sense". Base-form DLIG
demonstration: no aggregation across examples, no token-role labels, no
statistical claim — just the attribution map the reader inspects. Reads
--dlig (from wic.py). --panel_per_step shows an appendix view: same bars but
one row per denoising step (not averaged), to visualize how attribution
evolves over t, no claim.

Usage — stats (either or both axes):
  python -m helpers.analyze_wic \\
      --dlig outputs/wic/wic_dlig.jsonl \\
      --commit outputs/wic/wic_commitment.jsonl \\
      --out_dir outputs/wic/figs

Usage — main-text figures:
  python -m helpers.analyze_wic --depth_plot \\
      --dlig outputs/wic/wic_dlig.jsonl \\
      --depth_plot_out outputs/wic/figs/depth_centroid.png
  python -m helpers.analyze_wic --commit_plot \\
      --commit outputs/wic/wic_commitment.jsonl \\
      --commit_plot_out outputs/wic/figs/commit_trajectory.png

Usage — list panel candidates (hand-pick a vivid, concrete pivot word before
choosing --panel_idx; re-run after regenerating --dlig, since which idx
values are "correct" shifts between generation runs):
  python -m helpers.analyze_wic --list_candidates --dlig outputs/wic/wic_dlig.jsonl

Usage — panel mode (pick 2-3 examples, timestep-averaged panels over layers):
  python -m helpers.analyze_wic --panel --dlig outputs/wic/wic_dlig.jsonl \\
      --panel_idx 208 --panel_out_file outputs/wic/figs/correct_no_wall.png
  python -m helpers.analyze_wic --panel --dlig outputs/wic/wic_dlig.jsonl \\
      --panel_pick correct_no --panel_out_file outputs/wic/figs/correct_no.png

  # appendix: per-step evolution for one example
  python -m helpers.analyze_wic --panel --dlig outputs/wic/wic_dlig.jsonl \\
      --panel_idx 208 --panel_per_step \\
      --panel_out_file outputs/wic/figs/evolution_wall.png
"""

import os
import json
import argparse
import numpy as np


# ============================================================ IO + per-example

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def frac_deep(row, step_normalized=True):
    """Fraction of |attribution| mass in deep layers (>=20). From the DLIG jsonl.

    step_normalized=True (default, recommended): compute the deep-mass fraction
    WITHIN each denoising step, then average those fractions — so every step
    contributes equally regardless of its raw magnitude. This preempts the
    reviewer concern that steps with larger absolute attribution could dominate
    a naive cross-step sum. (WiC's frac_deep was the only metric in the paper that
    summed raw magnitude across steps; ProsQA averages per-step by construction
    and Infilling normalizes per-step via its `normalize` flag, so neither needs
    this.) Robustness: AUC 0.696 (raw sum) -> 0.663 (step-normalized); the pred-Yes
    > pred-No depth effect survives every normalization down to layer-centroid.

    step_normalized=False: legacy raw cross-step sum, kept for reproducibility."""
    if step_normalized:
        fr = []
        for sd in row["steps_data"]:
            per = {int(ls): np.abs(np.array(sc, float)).sum()
                   for ls, sc in sd["layers"].items()}
            tot = sum(per.values()) + 1e-12
            fr.append(sum(per[l] for l in per if l >= 20) / tot)
        return float(np.mean(fr)) if fr else None
    # legacy: sum over steps first, then take deep fraction
    acc, cnt = {}, {}
    for sd in row["steps_data"]:
        for ls, sc in sd["layers"].items():
            l = int(ls); v = np.abs(np.array(sc, float))
            acc[l] = acc.get(l, 0) + v; cnt[l] = cnt.get(l, 0) + 1
    per = {l: (acc[l] / cnt[l]).sum() for l in acc}
    tot = sum(per.values()) + 1e-12
    return sum(per[l] for l in per if l >= 20) / tot


def mean_depth(row, step_normalized=True, eps=1e-9):
    """Mass-weighted mean layer index (attribution centroid over layers) — the
    §5.1 'attribution depth'. Larger = attribution sits deeper in the stack.

    Steps whose TOTAL attribution mass is ~0 carry no depth information and are
    EXCLUDED, so they cannot bias the centroid toward layer 0. This matters on
    WiC: about a quarter of denoising steps are near-empty, and folding them in
    as centroid-0 would drag the mean shallow and inflate its variance. Returns
    None if an example has no surviving step (dropped downstream).

    step_normalized=True (default): centroid computed WITHIN each surviving step,
    then averaged, so every step contributes equally regardless of its absolute
    magnitude. step_normalized=False: legacy single cross-step mass-weighting."""
    if step_normalized:
        cent = []
        for sd in row["steps_data"]:
            per = {int(ls): np.abs(np.array(sc, float)).sum()
                   for ls, sc in sd["layers"].items()}
            tot = sum(per.values())
            if tot <= eps:
                continue
            cent.append(sum(l * per[l] for l in per) / tot)
        return float(np.mean(cent)) if cent else None
    # legacy: single cross-step mass-weighted centroid
    acc = {}
    for sd in row["steps_data"]:
        for ls, sc in sd["layers"].items():
            l = int(ls); acc[l] = acc.get(l, 0.0) + np.abs(np.array(sc, float)).sum()
    tot = sum(acc.values())
    return (sum(l * acc[l] for l in acc) / tot) if tot > eps else None


def report_empty_steps(rows, eps=1e-9):
    """Per-step + aggregate fraction of near-empty (total-mass<=eps) cells among
    parseable-pred examples — precisely the steps mean_depth() excludes from the
    centroid via `if tot <= eps: continue`. Regenerates the §5.1 depth footnote
    (~25% of steps near-empty, concentrated early) from the --dlig jsonl, so the
    reported numbers come from this script rather than a scratch computation.
    `rows` = loaded --dlig jsonl; only pred in {0,1} rows enter the depth analysis."""
    rows = [r for r in rows if r.get("pred") in (0, 1)]
    tot = {}; zero = {}
    for r in rows:
        for sd in r["steps_data"]:
            m = sum(np.abs(np.array(sc, float)).sum()
                    for sc in sd["layers"].values())
            s = sd["step"]
            tot[s] = tot.get(s, 0) + 1
            if m <= eps:
                zero[s] = zero.get(s, 0) + 1
    steps = sorted(tot)
    allz = sum(zero.get(s, 0) for s in steps); alln = sum(tot.values())
    print(f"[WiC empty-step report] eps={eps:g}, "
          f"n_examples(pred in 0/1)={len(rows)}")
    print(f"  {'step':>4} {'zero':>6} {'total':>6} {'pct':>6}")
    for s in steps:
        z = zero.get(s, 0)
        print(f"  {s:>4} {z:>6} {tot[s]:>6} {100*z/tot[s]:>5.1f}%")
    print(f"  {'ALL':>4} {allz:>6} {alln:>6} {100*allz/alln:>5.1f}%")
    pcts = [round(100 * zero.get(s, 0) / tot[s]) for s in steps]
    print(f"  footnote: steps {steps} -> {pcts}%  "
          f"(aggregate {100*allz/alln:.1f}%)")
    return {"per_step": {s: (zero.get(s, 0), tot[s]) for s in steps},
            "aggregate": (allz, alln)}


def commit_step(rec, conf=0.5):
    """Normalised denoising step (0..1) at which the model becomes CONFIDENT in
    its answer, and stays confident through the end. Confidence = raw softmax prob
    of the committed answer token (p_yes for a Yes commit, p_no for a No commit)
    crossing an absolute threshold `conf` — NOT the 2-way ratio, which fires as
    soon as the winner merely edges ahead and produced a degenerate step-0 spike.
    Larger = becomes confident later. Returns 1.0 if never confident.

    Takes the full record (needs p_yes AND p_no); committed side from final p_yes."""
    py = np.asarray(rec["p_yes"], float)
    pn = np.asarray(rec["p_no"], float)
    p2 = np.asarray(rec["p_yes_2way"], float)
    n = len(py)
    side = 1 if p2[-1] > 0.5 else 0
    conf_track = py if side == 1 else pn        # raw prob of the committed token
    on = conf_track >= conf
    for i in range(n):
        if on[i] and on[i:].all():
            return i / max(n - 1, 1)
    return 1.0


# ============================================================ stats primitives

def auc_mw(pos, neg):
    """AUC (= P(pos>neg)) and two-sided Mann-Whitney p (normal approx, tie-corr)."""
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


try:
    import diptest as _diptest
    _HAVE_DIP = True
except ImportError:
    _HAVE_DIP = False


def dip_test(x):
    """Hartigan dip test for unimodality via the `diptest` package. A hand-rolled
    version was tried and discarded (failed the unimodal sanity check)."""
    x = np.asarray(x, float)
    if not _HAVE_DIP:
        return np.nan, np.nan
    if len(x) < 4:
        return 0.0, 1.0
    d, p = _diptest.diptest(x)
    return float(d), float(p)


def _gauss_ll(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


def gmm_bic(x, seed=0, iters=200):
    """BIC for 1- and 2-component Gaussian fits (label-free). Returns (bic1, bic2,
    modes, weights). BIC over-splits skewed unimodal data — always read alongside
    the dip test, never alone."""
    x = np.asarray(x, float); n = len(x)
    # 1-component
    mu1, var1 = x.mean(), x.var() + 1e-6
    bic1 = -2 * np.sum(_gauss_ll(x, mu1, var1)) + 2 * np.log(n)
    # 2-component EM
    mu = np.quantile(x, [0.25, 0.75]).astype(float)
    var = np.array([x.var() + 1e-6] * 2); w = np.array([0.5, 0.5])
    for _ in range(iters):
        lp = np.stack([np.log(w[k] + 1e-12) + _gauss_ll(x, mu[k], var[k])
                       for k in range(2)], axis=1)
        lp -= lp.max(axis=1, keepdims=True)
        r = np.exp(lp); r /= r.sum(axis=1, keepdims=True)
        Nk = r.sum(axis=0) + 1e-12
        mu = (r * x[:, None]).sum(axis=0) / Nk
        var = (r * (x[:, None] - mu) ** 2).sum(axis=0) / Nk + 1e-6
        w = Nk / n
    ll = np.sum(np.log(np.sum(
        [w[k] * np.exp(_gauss_ll(x, mu[k], var[k])) for k in range(2)], axis=0) + 1e-12))
    bic2 = -2 * ll + 5 * np.log(n)
    return bic1, bic2, mu, w


# ============================================================ experiments

def experiment1(examples, axis_key, axis_name):
    """Yes-vs-No behavioural contrast on one axis, grouped by prediction."""
    predY = [e[axis_key] for e in examples if e["pred"] == 1 and e[axis_key] is not None]
    predN = [e[axis_key] for e in examples if e["pred"] == 0 and e[axis_key] is not None]
    auc, p, (n1, n2) = auc_mw(predY, predN)
    print(f"\n[Exp1 / {axis_name}]  predicted-Yes vs predicted-No")
    print(f"   predY n={n1} mean={np.nanmean(predY):.3f}   "
          f"predN n={n2} mean={np.nanmean(predN):.3f}")
    star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""
    print(f"   AUC={auc:.3f}  p={p:.4f}  {star}")


def experiment2(examples, axis_key, axis_name, out_dir):
    """Unsupervised bimodality within pred-Yes on one axis."""
    sub = [e for e in examples if e["pred"] == 1 and e[axis_key] is not None]
    v = np.array([e[axis_key] for e in sub], float)
    lab = np.array([e["label"] for e in sub])
    if len(v) < 4:
        print(f"\n[Exp2 / {axis_name}]  too few pred-Yes examples ({len(v)})")
        return
    d, pdip = dip_test(v)
    bic1, bic2, mu, w = gmm_bic(v)
    print(f"\n[Exp2 / {axis_name}]  bimodality within pred-Yes (n={len(v)})")
    if np.isnan(pdip):
        print("   dip: [diptest not installed — pip install diptest]")
    else:
        print(f"   dip={d:.4f}  p={pdip:.4f}  "
              f"({'REJECT unimodal' if pdip < 0.05 else 'no evidence vs unimodal'})")
    print(f"   BIC 1-comp={bic1:.1f}  2-comp={bic2:.1f}  "
          f"({'2-comp' if bic2 < bic1 else '1-comp'} preferred; "
          f"modes {mu[0]:.3f}/{mu[1]:.3f}, w {w[0]:.2f}/{w[1]:.2f})")
    verdict = ("bimodal" if (not np.isnan(pdip) and pdip < 0.05) else "unimodal")
    print(f"   -> verdict (dip-led): {verdict}")

    if out_dir:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(out_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(5, 3.2))
            bins = np.linspace(v.min(), v.max(), 24)
            ax.hist(v[lab == 1], bins=bins, alpha=0.6, color="C0", label="gold-Yes")
            ax.hist(v[lab == 0], bins=bins, alpha=0.6, color="C3", label="gold-No")
            ax.set_title(f"{axis_name} within pred-Yes  (dip p="
                         f"{pdip:.3f})" if not np.isnan(pdip) else axis_name)
            ax.set_xlabel(axis_name); ax.set_ylabel("count")
            ax.legend(fontsize=8, title="post-hoc (future work)")
            fig.tight_layout()
            fp = os.path.join(out_dir, f"bimodal_{axis_key}.png")
            fig.savefig(fp, dpi=150); plt.close(fig)
            print(f"   [fig] {fp}")
        except Exception as e:
            print(f"   [fig skipped: {e}]")


# ============================================================ main-text figures

# Okabe-Ito categorical pair (colour-blind-safe): Yes=blue, No=orange.
_BLUE, _ORANGE = "#0072B2", "#E69F00"


def _fig_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.8})


def plot_depth_centroid(rows, out_file, eps=1e-9):
    """AXIS A figure: per-example attribution centroid (mean layer) distribution,
    pred-Yes vs pred-No — violin + box + jittered points, grouped by each
    example's own predicted label. Renders the 'same depth' result. Also emits a
    vector .pdf beside a .png out_file."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yes, no = [], []
    for r in rows:
        if r.get("pred") not in (0, 1):
            continue
        c = mean_depth(r)                       # degenerate steps already excluded
        if c is None:
            continue
        (yes if r["pred"] == 1 else no).append(c)
    yes, no = np.array(yes), np.array(no)

    _fig_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    data, cols, pos = [yes, no], [_BLUE, _ORANGE], [1, 2]
    vp = ax.violinplot(data, positions=pos, widths=0.8, showextrema=False)
    for b, c in zip(vp["bodies"], cols):
        b.set_facecolor(c); b.set_alpha(0.22); b.set_edgecolor(c); b.set_linewidth(1.0)
    bp = ax.boxplot(data, positions=pos, widths=0.14, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.3),
                    whiskerprops=dict(color="0.35"), capprops=dict(color="0.35"),
                    boxprops=dict(linewidth=0.0))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor("white"); patch.set_edgecolor(c); patch.set_linewidth(1.2)
    rng = np.random.default_rng(0)
    for x, d, c in zip(pos, data, cols):
        ax.scatter(x - 0.30 + (rng.random(len(d)) - 0.5) * 0.10, d,
                   s=6, color=c, alpha=0.30, linewidths=0, zorder=1)
    for x, d, c in zip(pos, data, cols):
        ax.scatter([x], [d.mean()], marker="D", s=44, color=c,
                   edgecolor="black", linewidth=0.7, zorder=5)
        ax.annotate(f"{d.mean():.2f}", (x, d.mean()), xytext=(14, 0),
                    textcoords="offset points", va="center", fontsize=9, color=c)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"pred-Yes\n(n={len(yes)})", f"pred-No\n(n={len(no)})"])
    ax.set_ylabel("attribution centroid (mean layer)")
    ax.set_ylim(4.5, 9.5); ax.set_yticks(range(5, 10))
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    fig.savefig(out_file, dpi=200)
    if out_file.endswith(".png"):
        fig.savefig(out_file[:-4] + ".pdf")     # vector for LaTeX
    plt.close(fig)
    print(f"[SUCCESS] Saved {out_file}   "
          f"(Yes n={len(yes)} mean={yes.mean():.3f}, No n={len(no)} mean={no.mean():.3f})")


def plot_commit_trajectory(rows, out_file):
    """AXIS B figure: mean confidence in the committed answer (raw softmax prob of
    the model's own Yes/No token) across denoising steps, pred-Yes vs pred-No,
    with +-1 SEM bands and the 0.5 commit threshold marked. Grouped by each
    example's own predicted label. Renders the 'Yes commits earlier' result.
    Also emits a vector .pdf beside a .png out_file."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yes, no = [], []
    for r in rows:
        if r.get("pred") not in (0, 1):
            continue
        conf = np.asarray(r["p_yes"] if r["pred"] == 1 else r["p_no"], float)
        (yes if r["pred"] == 1 else no).append(conf)
    yes, no = np.array(yes), np.array(no)
    n = yes.shape[1]
    x = np.arange(n) / (n - 1)                   # normalized denoising step 0..1

    _fig_style()
    fig, ax = plt.subplots(figsize=(4.7, 3.4))
    for Tr, c, lab in [(yes, _BLUE, "pred-Yes"), (no, _ORANGE, "pred-No")]:
        m = Tr.mean(0); sem = Tr.std(0) / np.sqrt(len(Tr))
        ax.fill_between(x, m - sem, m + sem, color=c, alpha=0.22, linewidth=0)
        ax.plot(x, m, color=c, linewidth=2.0, marker="o", markersize=4,
                label=f"{lab} (n={len(Tr)})")
    ax.axhline(0.5, ls="--", color="0.4", linewidth=1.0)
    ax.text(0.02, 0.52, "commit threshold", ha="left", va="bottom",
            fontsize=8, color="0.4")
    ax.set_xlabel("denoising step (normalized)")
    ax.set_ylabel("confidence in committed answer")
    ax.set_ylim(0, 1); ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    fig.savefig(out_file, dpi=200)
    if out_file.endswith(".png"):
        fig.savefig(out_file[:-4] + ".pdf")      # vector for LaTeX
    plt.close(fig)
    cs_yes = np.array([commit_step(r) for r in rows if r.get("pred") == 1])
    cs_no = np.array([commit_step(r) for r in rows if r.get("pred") == 0])
    print(f"[SUCCESS] Saved {out_file}   "
          f"(commit_step Yes mean={cs_yes.mean():.3f}, No mean={cs_no.mean():.3f})")


# ============================================================ panel mode (per-example attribution bars)

def list_panel_candidates(rows, n=40):
    """Print candidate examples for --panel_idx, grouped by (correct,
    pred==label) -- i.e. the model's correct No (overrode its Yes-bias) and
    correct Yes populations. Prefer vivid, concrete pivot words (bank, bass,
    light, spring, wall, throw, channel, ...) over abstract ones (have, make)
    when hand-picking a qualitative example -- concrete senses make the
    attribution bars easier for a reader to sanity-check against the actual
    sentence. NOTE: which specific idx values are 'correct' shifts between
    generation runs (different gen_steps/target_steps/checkpoint change which
    examples the model gets right) -- re-run this after regenerating --dlig
    rather than reusing indices from an older run."""
    def show(tag, pred, label):
        xs = [r for r in rows if r["correct"] and r["label"] == label and r["pred"] == pred]
        print(f"\n=== {tag} ({len(xs)}) ===")
        for r in xs[:n]:
            print(f'{r["idx"]:4d}  {r["word"]:<16} -> {r["gen_text"][:25]!r}')
    print("Pick vivid pivot words (bank, bass, light, spring...) over abstract ones (have, make).")
    show("correct_No (overrode Yes-bias)", 0, 0)
    show("correct_Yes", 1, 1)


def _pick_example(rows, idx, pick):
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
        raise SystemExit(f"no example matches --panel_pick {pick}.")
    return cands[0]


def _signed_d_by_layer(row, step_filter=None):
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
    return t.replace("Ġ", "").replace("_", "").lstrip()


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


def plot_layer_panels(row, layers, out_file, per_step=False, min_frac=0.05, cols=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = row["input_tokens"]

    if not per_step:
        # MAIN: timestep-averaged, one panel per layer
        dbl = _signed_d_by_layer(row)
        layers = [l for l in layers if l in dbl]
        n = len(layers)
        cols = cols if cols is not None else min(4, n)
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
            dbl = _signed_d_by_layer(row, step_filter=st)
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


# ============================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlig", default="outputs/wic/wic_dlig.jsonl",
                    help="attribution jsonl (axis A: depth; also panel/depth-plot input)")
    ap.add_argument("--commit", default="outputs/wic/wic_commitment.jsonl",
                    help="commitment jsonl (axis B: commit_step; also commit-plot input)")
    ap.add_argument("--out_dir", default="outputs/wic/figs")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="axis B: absolute prob threshold for 'confident' commit")
    ap.add_argument("--depth_metric", choices=["centroid", "frac_deep"],
                    default="centroid",
                    help="axis A: 'centroid' = mass-weighted mean layer index "
                         "(attribution depth, §5.1); 'frac_deep' = share of mass "
                         "in layers >=20 (legacy)")
    ap.add_argument("--legacy_frac_deep", action="store_true",
                    help="axis A: use the raw cross-step sum (pre-normalization) "
                         "instead of the step-normalized metric")
    ap.add_argument("--empty_report", action="store_true",
                    help="print per-step near-empty-cell counts (the steps "
                         "mean_depth excludes from the centroid) from --dlig and "
                         "exit; regenerates the §5.1 depth footnote numbers")

    # --- main-text figures ---
    ap.add_argument("--depth_plot", action="store_true",
                    help="render the attribution-depth centroid figure (pred-Yes vs "
                         "pred-No) from --dlig and exit")
    ap.add_argument("--depth_plot_out", default="outputs/wic/figs/depth_centroid.png")
    ap.add_argument("--commit_plot", action="store_true",
                    help="render the commitment-trajectory figure (pred-Yes vs "
                         "pred-No) from --commit and exit")
    ap.add_argument("--commit_plot_out", default="outputs/wic/figs/commit_trajectory.png")

    # --- panel mode ---
    ap.add_argument("--panel", action="store_true",
                    help="run panel mode (per-example attribution bars) instead of the stats experiments")
    ap.add_argument("--list_candidates", action="store_true",
                    help="print candidate examples for --panel_idx, grouped by "
                         "correct_No (overrode Yes-bias) / correct_Yes, from "
                         "--dlig, and exit. Hand-pick a vivid, concrete pivot "
                         "word from the printed list rather than an abstract "
                         "one -- re-run this after regenerating --dlig, since "
                         "which idx values are 'correct' shifts between runs.")
    ap.add_argument("--list_n", type=int, default=40,
                    help="--list_candidates: max examples printed per group.")
    ap.add_argument("--panel_out_file", default=None,
                    help="required with --panel: where to save the panel figure")
    ap.add_argument("--panel_idx", type=int, default=None,
                    help="panel mode: example idx to plot; if unset use --panel_pick")
    ap.add_argument("--panel_pick", default="correct_no",
                    choices=["correct_no", "correct_yes", "any_correct", "any"],
                    help="panel mode: auto-pick population when --panel_idx not given")
    ap.add_argument("--panel_layers", type=int, nargs="+",
                    default=[0, 2, 4, 8, 12, 16, 18, 20, 22],
                    help="panel mode: which layers to panel. Default is "
                         "Figure 1's exact 9-layer set (DLIG_NeurIPS.pdf).")
    ap.add_argument("--panel_per_step", action="store_true",
                    help="panel mode appendix: show per-step evolution instead of averaging")
    ap.add_argument("--panel_min_frac", type=float, default=0.08,
                    help="panel mode: drop tokens with |attr| below this fraction "
                         "of the panel max (floor: keep >=3 largest regardless). "
                         "0.08 matches Figure 1's stated rule exactly.")
    ap.add_argument("--panel_cols", type=int, default=3,
                    help="panel mode: subplot grid columns. 3 matches Figure 1's "
                         "3x3 layout for the 9-layer default above.")
    args = ap.parse_args()

    if args.list_candidates:
        list_panel_candidates(load(args.dlig), n=args.list_n)
        return

    if args.panel:
        if not args.panel_out_file:
            raise SystemExit("--panel requires --panel_out_file")
        rows = load(args.dlig)
        row = _pick_example(rows, args.panel_idx, args.panel_pick)
        plot_layer_panels(row, args.panel_layers, args.panel_out_file,
                          per_step=args.panel_per_step, min_frac=args.panel_min_frac,
                          cols=args.panel_cols)
        return

    if args.empty_report:
        report_empty_steps(load(args.dlig))
        return

    if args.depth_plot:
        plot_depth_centroid(load(args.dlig), args.depth_plot_out)
        return

    if args.commit_plot:
        plot_commit_trajectory(load(args.commit), args.commit_plot_out)
        return

    # Build one example table keyed by idx, carrying both axes when available.
    depth_fn = mean_depth if args.depth_metric == "centroid" else frac_deep
    depthA_name = f"AXIS A: attribution depth [{args.depth_metric}]"
    ex = {}

    if os.path.exists(args.dlig):
        for r in load(args.dlig):
            ex.setdefault(r["idx"], {}).update(
                idx=r["idx"], word=r["word"], label=r["label"],
                pred=r["pred"],
                depth=depth_fn(r, step_normalized=not args.legacy_frac_deep))
        print(f"[info] axis A depth metric = {args.depth_metric}")
    else:
        print(f"[skip axis A] {args.dlig} not found")

    if os.path.exists(args.commit):
        for r in load(args.commit):
            d = ex.setdefault(r["idx"], {})
            d.update(idx=r["idx"], word=r.get("word"), label=r["label"], pred=r["pred"])
            d["commit_step"] = commit_step(r, conf=args.conf)
    else:
        print(f"[skip axis B] {args.commit} not found")

    examples = [e for e in ex.values() if e.get("pred") in (0, 1)]
    # ensure both keys exist (None if that axis's file was absent)
    for e in examples:
        e.setdefault("depth", None)
        e.setdefault("commit_step", None)

    n_dlig = sum(e["depth"] is not None for e in examples)
    n_com = sum(e["commit_step"] is not None for e in examples)
    print(f"[info] {len(examples)} examples with a parseable prediction "
          f"(axis A depth: {n_dlig}, axis B commit_step: {n_com})")

    have_A = n_dlig > 0
    have_B = n_com > 0

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 — Yes vs No behaviour (grouped by prediction)")
    print("=" * 60)
    if have_A:
        experiment1(examples, "depth", depthA_name)
    if have_B:
        experiment1(examples, "commit_step", "AXIS B: commitment timing")

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 — hidden structure within pred-Yes (unsupervised)")
    print("=" * 60)
    if have_A:
        experiment2(examples, "depth", depthA_name, args.out_dir)
    if have_B:
        experiment2(examples, "commit_step", "AXIS B: commitment timing", args.out_dir)

    print("\nExp1 groups by the model's predicted answer (behavioural), not gold.")
    print("Exp2 is label-free; trust the dip verdict over BIC. Gold overlay on the "
          "histograms is post-hoc colour only — future-work hypothesis, not a claim.")


if __name__ == "__main__":
    main()