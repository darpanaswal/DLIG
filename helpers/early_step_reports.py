"""
Empty-step reporters for the WiC depth footnote (§5.1) and the infill footnote (§7).

Both regenerate, from your own scripts, the exclusion counts that were previously
computed only in a scratch analysis:

  WiC   footnote: ~25% of denoising steps near-empty, concentrated early
                  (steps 1,3,5,7,9,11 -> ~78,44,20,8,2,1%).
  Infill footnote: surviving n per step (n=544 at t=1, n>=981 at t>=5), plus a
                  cross-tab proving n_scoreable==0 <=> total-mass<=eps on this data
                  (the empirical fact that licenses "for the same reason as §5").

Verified against wic_dlig.jsonl (491 parseable-pred rows) and diffugpt_self.jsonl
(1000 stories): WiC per-step 77.4/44.2/20.8/7.9/2.0/0.8%, aggregate 25.5%; infill
surviving n exactly 544/875/981/997/1000/1000, cross-tab off-diagonals both 0.

--------------------------------------------------------------------------------
PART A — paste into analyze_wic.py (the current version, the one with mean_depth).
Put report_empty_steps_wic() right after mean_depth(), since it reports on exactly
the cells mean_depth's `if tot <= eps: continue` drops.
--------------------------------------------------------------------------------
"""

from collections import defaultdict
import numpy as np


def report_empty_steps_wic(rows, eps=1e-9):
    """Per-step + aggregate fraction of near-empty (total-mass<=eps) cells among
    parseable-pred examples — precisely the steps mean_depth() excludes from the
    centroid. Reproduces the §5.1 depth footnote. `rows` = loaded --dlig jsonl."""
    rows = [r for r in rows if r.get("pred") in (0, 1)]
    tot = defaultdict(int); zero = defaultdict(int)
    for r in rows:
        for sd in r["steps_data"]:
            m = sum(np.abs(np.array(sc, float)).sum()
                    for sc in sd["layers"].values())
            s = sd["step"]; tot[s] += 1
            if m <= eps:
                zero[s] += 1
    steps = sorted(tot)
    allz = sum(zero.values()); alln = sum(tot.values())
    print(f"[WiC empty-step report] eps={eps:g}, "
          f"n_examples(pred in 0/1)={len(rows)}")
    print(f"  {'step':>4} {'zero':>6} {'total':>6} {'pct':>6}")
    for s in steps:
        print(f"  {s:>4} {zero[s]:>6} {tot[s]:>6} {100*zero[s]/tot[s]:>5.1f}%")
    print(f"  {'ALL':>4} {allz:>6} {alln:>6} {100*allz/alln:>5.1f}%")
    pcts = [round(100 * zero[s] / tot[s]) for s in steps]
    print(f"  footnote: steps {steps} -> {pcts}%  "
          f"(aggregate {100*allz/alln:.1f}%)")
    return {"per_step": {s: (zero[s], tot[s]) for s in steps},
            "aggregate": (allz, alln)}


# --- argparse hook for analyze_wic.py: add near the other flags ---
#     ap.add_argument("--empty_report", action="store_true",
#                     help="print per-step near-empty-cell counts (the steps "
#                          "mean_depth excludes) from --dlig and exit")
#
# --- dispatch: add in main(), before the stats table is built ---
#     if args.empty_report:
#         report_empty_steps_wic(load(args.dlig))
#         return


# ==============================================================================
# PART B — paste into analyze_infill.py. Put report_empty_steps_infill() near
# aggregate_infill(); it reports on the same cells aggregate_infill() skips via
# `if sd.get("skipped") or sd.get("n_scoreable") == 0: continue`.
# ==============================================================================

import json as _json
import os as _os


def report_empty_steps_infill(input_file, eps=1e-9):
    """Per-step surviving-n (n_scoreable>0) vs dropped (F_t==0) for the
    self-generated framing — the cells aggregate_infill() skips. Reproduces the
    §7 footnote sample sizes. Also cross-tabs n_scoreable==0 against
    total-mass<=eps to confirm the two criteria coincide (the fact that makes the
    infill exclusion 'the same reason as' the WiC one)."""
    if not _os.path.exists(input_file):
        print(f"[ERROR] missing: {input_file}")
        return None
    surv = defaultdict(int); dropp = defaultdict(int)
    ct = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for line in open(input_file):
        if not line.strip():
            continue
        row = _json.loads(line)
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
            (dropp if sc0 else surv)[s] += 1
    steps = sorted(set(surv) | set(dropp))
    print(f"[infill empty-step report] eps={eps:g}")
    print(f"  {'step':>4} {'surv':>6} {'dropped':>7} {'total':>6}")
    for s in steps:
        print(f"  {s:>4} {surv[s]:>6} {dropp[s]:>7} {surv[s]+dropp[s]:>6}")
    print(f"  footnote: surviving n per step -> "
          f"{{{', '.join(f't{s}:{surv[s]}' for s in steps)}}}")
    print(f"  cross-tab (n_scoreable==0 vs mass<=eps): "
          f"both0={ct[(True,True)]}, sc0&mass>0={ct[(True,False)]}, "
          f"sc>0&mass0={ct[(False,True)]}, both>0={ct[(False,False)]}")
    agree = ct[(True, False)] == 0 and ct[(False, True)] == 0
    print(f"  -> criteria {'COINCIDE exactly' if agree else 'DIVERGE'} on this data")
    return {"surviving": dict(surv), "dropped": dict(dropp), "crosstab": ct}


# --- argparse hook for analyze_infill.py: add near the other flags ---
#     parser.add_argument("--empty_report", action="store_true",
#                         help="print per-step surviving/dropped cell counts "
#                              "(F_t==0 exclusion) + zero-scoreable/zero-mass "
#                              "cross-tab, then exit")
#
# --- dispatch: add in main() right after input_file is resolved ---
#     if args.empty_report:
#         report_empty_steps_infill(input_file)
#         return