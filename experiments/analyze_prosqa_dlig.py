#!/usr/bin/env python3
# experiments/analyze_prosqa_dlig.py
r"""
analyze_prosqa_dlig.py — analyses A-D over prosqa_contrastive_dlig.py output.

Per example, per (layer, step), let s+[i], s-[i] be per-token DLIG toward
y+ (gold) and y- (wrong option). Define

    d[i]      = s+[i] - s-[i]                    # contrastive attribution
    d_beh[i]  = d[i]  if group == success        # behavior-aligned direction:
              = -d[i] if group == fail           # fail => model preferred y-

Token sets come from span_ids -> graph labels:
    GOLD  = tokens of gold-path edges + root fact
    WRONG = tokens of edges on the root->wrong-option path (when it exists)
    CHAIN = tokens of edges the model itself cited in `gen`
    FACT  = all edge tokens + root (denominator; question span EXCLUDED by
            default because it contains the option words themselves)

(A) Path precision.
    prec = sum_{i in GOLD} max(d[i], 0) / sum_{i in FACT} max(d[i], 0)
    chance = |GOLD| / |FACT|  (token counts). Report prec - chance per example
    (success group), Wilcoxon signed-rank vs 0, plus an empirical null from
    random edge subsets with the same edge count as the gold path.

(B) Failure faithfulness (fail group, behavior-aligned d_beh).
    chain_mass vs gold_mass: normalized positive-mass fraction on CHAIN\GOLD
    vs GOLD\CHAIN tokens (disjoint sets, per-token normalized to remove set
    size effects). Faithful attribution => mass tracks CHAIN, not GOLD.

(C) Reliance predicts correctness.
    Per example, total mass M = mean_i |d[i]| over FACT tokens (per-token to
    kill length confound), pooled over (layer, step). AUC(success vs fail)
    = Mann-Whitney U / (n1*n2); also per-step AUC curve (extends the
    ROCStories mass->ROUGE result to discrete correctness).

(D) Hop-resolved profile + localization.
    Per gold-path hop h (root fact = hop 0), mean per-token d_beh among
    successes, normalized within example by mean over hops. Chain-following
    => mass spread along hops; shortcut => mass only at h=0 and h=k.
    Plus (layer, step) heatmap of gold-mass fraction.

Stats: scipy when available, exact-normal fallbacks otherwise.
"""

import os
import json
import argparse
import math
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ---------------- stats helpers ----------------

def wilcoxon_vs_zero(x):
    """Wilcoxon signed-rank of x vs 0. Returns (stat, p). Normal fallback."""
    x = np.asarray(x, float)
    x = x[x != 0]
    if len(x) < 10:
        return np.nan, np.nan
    if HAVE_SCIPY:
        s, p = sps.wilcoxon(x)
        return float(s), float(p)
    r = np.argsort(np.argsort(np.abs(x))) + 1.0
    w = np.sum(r[x > 0])
    n = len(x)
    mu, sd = n * (n + 1) / 4, np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sd
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / np.sqrt(2))))
    return float(w), float(p)


def mannwhitney_auc(a, b):
    """AUC that a random success (a) outranks a random fail (b), + p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    if HAVE_SCIPY:
        u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
        return float(u / (len(a) * len(b))), float(p)
    ranks = np.argsort(np.argsort(np.concatenate([a, b]))) + 1.0
    ra = ranks[: len(a)].sum()
    u = ra - len(a) * (len(a) + 1) / 2
    n1, n2 = len(a), len(b)
    mu, sd = n1 * n2 / 2, np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mu) / sd
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / np.sqrt(2))))
    return float(u / (n1 * n2)), float(p)


def logistic_irls(X, y, n_iter=50, tol=1e-8):
    """
    Logistic regression via IRLS (Newton-Raphson), no external deps.
    Model:  P(y=1 | x) = sigmoid(b0 + x . b)
    Update: b <- b + (X^T W X)^{-1} X^T (y - p),  W = diag(p (1 - p))
    Returns (beta, wald_p) with columns standardized; beta[0] = intercept.
    Wald p per coefficient: z = b_j / SE_j,  SE = sqrt(diag((X^T W X)^{-1})).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(y)), (X - mu) / sd])
    b = np.zeros(Xs.shape[1])
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-Xs @ b))
        W = p * (1 - p)
        H = Xs.T @ (Xs * W[:, None]) + 1e-9 * np.eye(Xs.shape[1])
        step = np.linalg.solve(H, Xs.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < tol:
            break
    p = 1.0 / (1.0 + np.exp(-Xs @ b))
    W = p * (1 - p)
    H = Xs.T @ (Xs * W[:, None]) + 1e-9 * np.eye(Xs.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    z = b / se
    wald_p = np.array([2 * (1 - 0.5 * (1 + math.erf(abs(zz) / np.sqrt(2))))
                       for zz in z])
    return b, wald_p


def spearman_r(a, b):
    a = np.argsort(np.argsort(np.asarray(a, float)))
    b = np.argsort(np.argsort(np.asarray(b, float)))
    if len(a) < 3:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ---------------- loading ----------------

def load_graph(path):
    g = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                g[r["idx"]] = r
    return g


def token_sets(span_ids, spans_by_id):
    """Index sets over the kept prompt tokens."""
    gold, wrong, chain, fact, quest = set(), set(), set(), set(), set()
    for i, sid in enumerate(span_ids):
        if sid < 0 or sid not in spans_by_id:
            continue
        sp = spans_by_id[sid]
        if sp["kind"] == "question":
            quest.add(i)
            continue
        fact.add(i)  # edges + root
        if sp["on_gold_path"]:
            gold.add(i)
        if sp["on_wrong_path"]:
            wrong.add(i)
        if sp["in_model_chain"]:
            chain.add(i)
    return gold, wrong, chain, fact, quest


def hop_of_token(span_ids, spans_by_id):
    return [spans_by_id[s]["hop"] if (s >= 0 and s in spans_by_id
                                      and spans_by_id[s]["kind"] != "question"
                                      and spans_by_id[s]["on_gold_path"])
            else -1
            for s in span_ids]


def pos_mass(d, idx):
    if not idx:
        return 0.0
    return float(np.sum(np.maximum(d[list(idx)], 0.0)))


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlig_file", required=True,
                    help="jsonl from prosqa_contrastive_dlig.py (cat shards first)")
    ap.add_argument("--graph_labels", required=True)
    ap.add_argument("--out_dir", default="outputs/prosqa")
    ap.add_argument("--n_null", type=int, default=200,
                    help="random edge subsets per example for the (A) null")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    graph = load_graph(args.graph_labels)

    # accumulators
    prec_gap = []                      # (A) prec - chance, success, pooled
    prec_gap_null = []                 # (A) random-subset null gaps
    prec_by_ls = defaultdict(list)     # (A/D) (layer, step) -> gold-mass fraction
    fail_chain, fail_gold = [], []     # (B) per-token masses, fail group
    succ_chain_ctrl = []               # (B) control: successes chain==gold mostly
    mass_by_group = defaultdict(list)  # (C) pooled per-token |d| mass
    mass_by_step = defaultdict(lambda: defaultdict(list))  # (C) step -> group
    cov_rows = []                      # (C) per-example (label, mass, k, n_edges)
    fail_chain_vs_rest = []            # (B) all fails: chain minus non-chain mass
    succ_gold_vs_rest = []             # (B) mirror control on successes
    hop_profiles = defaultdict(list)   # (D) k_gold -> list of per-hop vectors
    n_used = 0

    with open(args.dlig_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            g = graph.get(rec["idx"])
            if g is None or not rec["steps_data"]:
                continue
            spans_by_id = {s["id"]: s for s in g["spans"]}
            span_ids = rec["span_ids"]
            GOLD, WRONG, CHAIN, FACT, _Q = token_sets(span_ids, spans_by_id)
            if not GOLD or not FACT:
                continue
            hops = np.array(hop_of_token(span_ids, spans_by_id))
            sign = 1.0 if rec["group"] == "success" else -1.0
            k = rec["k_gold"]

            # pooled contrastive attribution over (layer, step); also per (l, s)
            d_pool = None
            step_layer_mass = defaultdict(list)   # step -> [per-layer masses]
            for sd in rec["steps_data"]:
                step = sd["step"]
                for layer, ts in sd["layers"].items():
                    d = np.asarray(ts["plus"], float) - np.asarray(ts["minus"], float)
                    d_pool = d if d_pool is None else d_pool + d

                    pg = pos_mass(d, GOLD)
                    pf = pos_mass(d, FACT)
                    if pf > 0:
                        prec_by_ls[(int(layer), step)].append(pg / pf)

                    step_layer_mass[step].append(np.mean(np.abs(d[list(FACT)])))
            # ONE value per (example, step): mean over layers. Appending each
            # layer separately inflates n by 12x and invalidates the per-step
            # Mann-Whitney p-values (pseudo-replication).
            for step, ms in step_layer_mass.items():
                mass_by_step[step][rec["group"]].append(float(np.mean(ms)))

            if d_pool is None:
                continue
            n_used += 1
            d = d_pool
            d_beh = sign * d

            # ---- (A) precision vs chance + random null (success only) ----
            # ---- (A/E) gold-path precision, gold-direction d, ALL examples ----
            # (E) asks whether WHERE the mass sits predicts correctness, so the
            # precision is computed identically for successes and fails: always
            # in the gold direction d = s+ - s-, never behavior-aligned.
            pf_all = pos_mass(d, FACT)
            ex_prec_gap = np.nan
            if pf_all > 0:
                ex_prec_gap = (pos_mass(d, GOLD) / pf_all
                               - len(GOLD) / len(FACT))

            if rec["group"] == "success":
                pf = pos_mass(d, FACT)
                if pf > 0:
                    prec = pos_mass(d, GOLD) / pf
                    chance = len(GOLD) / len(FACT)
                    prec_gap.append(prec - chance)

                    # null: random edge subsets, same edge count as gold path
                    edge_ids = [s["id"] for s in g["spans"] if s["kind"] == "edge"]
                    gold_edge_ids = {s["id"] for s in g["spans"]
                                     if s["kind"] == "edge" and s["on_gold_path"]}
                    n_pick = len(gold_edge_ids)
                    if 0 < n_pick < len(edge_ids):
                        tok_by_edge = defaultdict(set)
                        for i, sid in enumerate(span_ids):
                            if sid in set(edge_ids):
                                tok_by_edge[sid].add(i)
                        for _ in range(args.n_null):
                            pick = rng.choice(edge_ids, size=n_pick, replace=False)
                            idx = set().union(*[tok_by_edge[e] for e in pick]) if n_pick else set()
                            if idx:
                                prec_gap_null.append(
                                    pos_mass(d, idx) / pf - len(idx) / len(FACT))

            # ---- (B) failure faithfulness ----
            chain_only = CHAIN - GOLD
            gold_only = GOLD - CHAIN
            if rec["group"] == "fail" and chain_only and gold_only:
                # per-token positive behavior-aligned mass
                fail_chain.append(pos_mass(d_beh, chain_only) / len(chain_only))
                fail_gold.append(pos_mass(d_beh, gold_only) / len(gold_only))
            if rec["group"] == "success" and chain_only and gold_only:
                succ_chain_ctrl.append(
                    pos_mass(d_beh, chain_only) / len(chain_only)
                    - pos_mass(d_beh, gold_only) / len(gold_only))

            # all-fails signed measure (uses every fail with a parsed chain,
            # not only those where CHAIN and GOLD partly disagree): per-token
            # behavior-aligned positive mass on CHAIN minus on FACT \ CHAIN.
            rest = FACT - CHAIN
            if rec["group"] == "fail" and CHAIN and rest:
                fail_chain_vs_rest.append(
                    pos_mass(d_beh, CHAIN) / len(CHAIN)
                    - pos_mass(d_beh, rest) / len(rest))
            # mirror control on successes: GOLD vs FACT \ GOLD (should be > 0
            # by (A); confirms the measure itself behaves).
            rest_g = FACT - GOLD
            if rec["group"] == "success" and rest_g:
                succ_gold_vs_rest.append(
                    pos_mass(d_beh, GOLD) / len(GOLD)
                    - pos_mass(d_beh, rest_g) / len(rest_g))

            # ---- (C) pooled mass ----
            ex_mass = float(np.mean(np.abs(d[list(FACT)])))
            mass_by_group[rec["group"]].append(ex_mass)
            cov_rows.append((1.0 if rec["group"] == "success" else 0.0,
                             ex_mass, float(k), float(rec["n_edges"]),
                             ex_prec_gap))

            # ---- (D) hop profile (success only) ----
            if rec["group"] == "success" and k >= 2:
                prof = np.full(k + 1, np.nan)
                for h in range(0, k + 1):
                    idx = np.where(hops == h)[0]
                    if len(idx):
                        prof[h] = np.mean(d_beh[idx])
                # normalize by mean |prof|, not |mean prof|: signed hop values
                # can cancel, making |mean| ~ 0 and exploding the ratio (the
                # unstable k=5 profile in the first run).
                mu = np.nanmean(np.abs(prof))
                if np.isfinite(mu) and mu > 1e-12:
                    hop_profiles[k].append(prof / mu)

    print(f"[ANALYZE] {n_used} examples used.")

    # =========================== report ===========================
    lines = [f"ProsQA contrastive DLIG analysis  (n={n_used})", "=" * 60]

    # (A)
    w, p = wilcoxon_vs_zero(prec_gap)
    null_mu = np.mean(prec_gap_null) if prec_gap_null else np.nan
    null_hi = np.percentile(prec_gap_null, 95) if prec_gap_null else np.nan
    lines += ["", "(A) Path precision (success group, pooled layers/steps)",
              f"    mean prec-chance gap : {np.mean(prec_gap):+.4f}  "
              f"(n={len(prec_gap)}; Wilcoxon p={p:.2e})",
              f"    random-edge null gap : mean {null_mu:+.4f}, 95th pct {null_hi:+.4f}"]

    # (B)
    auc_b, p_b = mannwhitney_auc(fail_chain, fail_gold)
    # PAIRED test on the same subset: chain\gold and gold\chain come from the
    # SAME example, so the signed-rank of the per-example difference is the
    # correct (and more powerful) test; unpaired MW discards the pairing.
    diffs = np.asarray(fail_chain) - np.asarray(fail_gold)
    w_pair, p_pair = wilcoxon_vs_zero(diffs)
    w_all, p_all = wilcoxon_vs_zero(fail_chain_vs_rest)
    w_sg, p_sg = wilcoxon_vs_zero(succ_gold_vs_rest)
    lines += ["", "(B) Failure faithfulness (fail group, behavior-aligned d)",
              f"    per-token mass  chain\\gold: {np.mean(fail_chain):+.4f}   "
              f"gold\\chain: {np.mean(fail_gold):+.4f}  (n={len(fail_chain)})",
              f"    paired Wilcoxon (chain\\gold - gold\\chain): "
              f"mean {np.mean(diffs) if len(diffs) else np.nan:+.4f}  p={p_pair:.2e}",
              f"    [unpaired ref] P(chain > gold) AUC={auc_b:.3f}  p={p_b:.2e}",
              f"    all-fails CHAIN vs FACT\\CHAIN (n={len(fail_chain_vs_rest)}): "
              f"mean {np.mean(fail_chain_vs_rest) if fail_chain_vs_rest else np.nan:+.4f}"
              f"  Wilcoxon p={p_all:.2e}",
              f"    success mirror GOLD vs FACT\\GOLD (n={len(succ_gold_vs_rest)}): "
              f"mean {np.mean(succ_gold_vs_rest) if succ_gold_vs_rest else np.nan:+.4f}"
              f"  Wilcoxon p={p_sg:.2e}",
              f"    success control (chain-gold gap): "
              f"{np.mean(succ_chain_ctrl) if succ_chain_ctrl else np.nan:+.4f}"]

    # (C)
    auc_c, p_c = mannwhitney_auc(mass_by_group["success"], mass_by_group["fail"])
    cov = np.asarray(cov_rows)          # cols: label, mass, k, n_edges, prec_gap
    lines += ["", "(C) Context-reliance predicts correctness",
              f"    per-token |d| mass  success: {np.mean(mass_by_group['success']):.4f}"
              f"   fail: {np.mean(mass_by_group['fail']):.4f}",
              f"    AUC(success vs fail) = {auc_c:.3f}  p={p_c:.2e}"]
    if len(cov):
        # difficulty confound: is mass just tracking problem size?
        lines.append(f"    confound check  Spearman(mass, k_gold)="
                     f"{spearman_r(cov[:, 1], cov[:, 2]):+.3f}   "
                     f"Spearman(mass, n_edges)="
                     f"{spearman_r(cov[:, 1], cov[:, 3]):+.3f}")
        # k-matched AUC: compare success vs fail WITHIN each path length
        lines.append("    k-matched AUC (mass -> correctness within k):")
        num, den = 0.0, 0.0
        for kk in sorted(set(cov[:, 2])):
            sel = cov[:, 2] == kk
            a = cov[sel & (cov[:, 0] == 1), 1]
            bmask = cov[sel & (cov[:, 0] == 0), 1]
            if len(a) >= 5 and len(bmask) >= 5:
                akk, pkk = mannwhitney_auc(a, bmask)
                w = len(a) + len(bmask)
                num += akk * w
                den += w
                lines.append(f"      k={int(kk)}: AUC={akk:.3f}  p={pkk:.2e}  "
                             f"(n={len(a)}+{len(bmask)})")
        if den > 0:
            lines.append(f"      weighted avg AUC = {num / den:.3f}")
        # logistic: correctness ~ mass + k_gold + n_edges (standardized).
        # beta_mass with controls in = the confound-adjusted effect.
        beta, wp = logistic_irls(cov[:, 1:4], cov[:, 0])
        lines.append("    logistic  correctness ~ mass + k_gold + n_edges "
                     "(standardized):")
        for name, bb, pp in zip(["mass", "k_gold", "n_edges"], beta[1:], wp[1:]):
            lines.append(f"      beta[{name}] = {bb:+.3f}  Wald p={pp:.2e}")
    lines.append("    per-step AUC (per-example, layer-averaged):")
    step_aucs = {}
    for step in sorted(mass_by_step):
        a, p_s = mannwhitney_auc(mass_by_step[step]["success"],
                                 mass_by_step[step]["fail"])
        step_aucs[step] = a
        lines.append(f"      t={step:2d}: AUC={a:.3f}  p={p_s:.2e}")

    # (E) Location vs magnitude: does WHERE the mass sits predict correctness?
    # AUC on the gold-direction precision gap, then a joint logistic
    #   correctness ~ precision + mass  (standardized)
    # Prediction from A+C: beta[precision] > 0 while beta[mass] < 0 --
    # correctness tracks placement of attribution, not amount.
    auc_e, p_e, prec_s, prec_f = np.nan, np.nan, [], []
    if len(cov):
        val = np.isfinite(cov[:, 4])
        prec_s = cov[val & (cov[:, 0] == 1), 4]
        prec_f = cov[val & (cov[:, 0] == 0), 4]
        auc_e, p_e = mannwhitney_auc(prec_s, prec_f)
        lines += ["", "(E) Attribution LOCATION predicts correctness",
                  f"    precision gap  success: {np.mean(prec_s):+.4f}   "
                  f"fail: {np.mean(prec_f):+.4f}",
                  f"    AUC(success vs fail) = {auc_e:.3f}  p={p_e:.2e}"]
        beta_e, wp_e = logistic_irls(cov[val][:, [4, 1]], cov[val, 0])
        lines.append("    logistic  correctness ~ precision + mass "
                     "(standardized):")
        for name, bb, pp in zip(["precision", "mass"], beta_e[1:], wp_e[1:]):
            lines.append(f"      beta[{name}] = {bb:+.3f}  Wald p={pp:.2e}")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out_dir, "prosqa_dlig_report.txt"), "w") as f:
        f.write(report + "\n")

    # =========================== plots ===========================
    # (A) gap histogram vs null
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(prec_gap, bins=30, alpha=0.7, label="gold path", density=True)
    if prec_gap_null:
        ax.hist(prec_gap_null, bins=30, alpha=0.5, label="random-edge null",
                density=True)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("positive-mass precision  -  chance")
    ax.set_title("(A) DLIG mass concentrates on the gold reasoning path")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "A_path_precision.png"), dpi=200)

    # (B) fail-group paired masses
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([fail_chain, fail_gold], tick_labels=["model chain\\gold",
                                                     "gold\\model chain"])
    ax.set_ylabel("per-token behavior-aligned positive mass")
    ax.set_title(f"(B) Failures: attribution tracks the model's own chain\n"
                 f"AUC={auc_b:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "B_failure_faithfulness.png"), dpi=200)

    # (C) mass violin + per-step AUC
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].violinplot([mass_by_group["success"], mass_by_group["fail"]],
                       showmeans=True)
    axes[0].set_xticks([1, 2], ["success", "fail"])
    axes[0].set_ylabel("mean per-token |dDLIG|")
    axes[0].set_title(f"(C) Reliance vs correctness  AUC={auc_c:.3f}")
    ss = sorted(step_aucs)
    axes[1].plot(ss, [step_aucs[s] for s in ss], marker="o")
    axes[1].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("denoising step t")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("per-step AUC (mass -> correctness)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "C_mass_predicts_correctness.png"), dpi=200)

    # (D) hop profiles per k + (layer, step) heatmap of gold-mass fraction
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in sorted(hop_profiles):
        arr = np.array(hop_profiles[k])
        if len(arr) < 5:
            continue
        axes[0].errorbar(range(k + 1), np.nanmean(arr, 0),
                         yerr=np.nanstd(arr, 0) / np.sqrt(len(arr)),
                         marker="o", label=f"k={k} (n={len(arr)})")
    axes[0].set_xlabel("hop (0 = root fact, k = answer edge)")
    axes[0].set_ylabel("normalized d_beh per token")
    axes[0].set_title("(D) Hop-resolved attribution: chain vs shortcut")
    axes[0].legend(fontsize=7)

    layers = sorted({l for (l, s) in prec_by_ls})
    steps = sorted({s for (l, s) in prec_by_ls})
    H = np.full((len(layers), len(steps)), np.nan)
    for (l, s), v in prec_by_ls.items():
        H[layers.index(l), steps.index(s)] = np.mean(v)
    im = axes[1].imshow(H, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_xticks(range(len(steps)), steps)
    axes[1].set_yticks(range(len(layers)), layers)
    axes[1].set_xlabel("denoising step t")
    axes[1].set_ylabel("layer")
    axes[1].set_title("gold-mass fraction by (layer, step)")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "D_hops_and_localization.png"), dpi=200)

    # (E) precision-by-group violin
    if len(prec_s) and len(prec_f):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.violinplot([prec_s, prec_f], showmeans=True)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xticks([1, 2], ["success", "fail"])
        ax.set_ylabel("gold-path precision  -  chance")
        ax.set_title(f"(E) Attribution location predicts correctness\n"
                     f"AUC={auc_e:.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir,
                                 "E_precision_predicts_correctness.png"), dpi=200)

    print(f"[ANALYZE] plots + report -> {args.out_dir}")


if __name__ == "__main__":
    main()