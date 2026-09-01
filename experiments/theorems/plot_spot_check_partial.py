#!/usr/bin/env python3
"""
plot_spot_check_partial.py — bar chart of partial-vs-full-path DLIG rel_diff per
layer, from a saved spot_check_partial.py log.

Usage:
    python -m experiments.theorems.spot_check_partial ... | tee runs/spot_check_partial.txt
    python -m experiments.theorems.plot_spot_check_partial \
        --log runs/spot_check_partial.txt \
        --out ../DLIG_NeurIPS/latex/completeness/spot_check_partial.png

Parses lines of the form (verbatim stdout of spot_check_partial.py):
    layer   max_abs_diff     rel_diff  verdict
     0        1.234e-05    5.678e-06  OK
     2        ...
"""
import re
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROW_RE = re.compile(r"^\s*(\S+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+(OK|\*\*\*.*)$")


def parse_log(path: pathlib.Path):
    rows = []
    for line in path.read_text().splitlines():
        m = ROW_RE.match(line)
        if m:
            layer, abs_diff, rel_diff, verdict = m.groups()
            rows.append((layer, float(abs_diff), float(rel_diff), verdict.startswith("OK")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=1e-2,
                     help="Pass/fail line from spot_check_partial.py (bf16 roundoff floor).")
    args = ap.parse_args()

    rows = parse_log(pathlib.Path(args.log))
    if not rows:
        raise SystemExit(f"No 'layer max_abs_diff rel_diff verdict' rows parsed from {args.log}")

    layers = [r[0] for r in rows]
    rel = [r[2] for r in rows]
    ok = [r[3] for r in rows]
    colors = ["#4C72B0" if o else "#C44E52" for o in ok]

    fig, ax = plt.subplots(figsize=(0.6 * len(layers) + 2, 4))
    ax.bar(range(len(layers)), rel, color=colors)
    ax.axhline(args.threshold, color="black", linestyle="--", linewidth=1,
               label=f"threshold ({args.threshold:.0e})")
    ax.set_yscale("log")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_xlabel("layer $\\ell$")
    ax.set_ylabel("rel. diff (partial-forward vs.\\ full-path)")
    ax.set_title("Partial-forward spot check: per-layer exactness")
    ax.legend()
    fig.tight_layout()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[SAVED] {out_path}")
    print(f"[SUMMARY] worst rel_diff = {max(rel):.3e} at layer {layers[rel.index(max(rel))]}")


if __name__ == "__main__":
    main()
