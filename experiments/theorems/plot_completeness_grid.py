#!/usr/bin/env python3
"""
plot_completeness_grid.py — turn the full-grid verify_completeness.sh logs
(runs/verify_completeness/grid_layer*.txt) into the two heatmaps (abs err,
rel err) that replace Table 1 in Appendix A.1.1 of the NeurIPS paper.

Usage:
    python -m experiments.theorems.plot_completeness_grid \
        --log_dir runs/verify_completeness \
        --out_dir ../DLIG_NeurIPS/latex/completeness \
        --m 1000

Expects one file per grid layer, named grid_layer{L}.txt, each containing
blocks of the form (verbatim stdout of verify_completeness.py):

    ================================================================
    STEP t=<step>   layer=<layer>
    ================================================================
    ...
    [B/C] completeness:  sum_{i,j} DLIG (ALL positions)  vs  deltaF
        m       sumDLIG          deltaF      abs_err      rel_err
         200   ...            ...           1.2e-03      1.5e-03
        1000   ...            ...           4.8e-04      5.8e-04

Pulls the row for --m (default 1000, the largest m in GRID_M_LIST) at every
(layer, step) cell and renders it as a heatmap, layers on the y-axis and
denoising steps on the x-axis, annotated with the numeric value in each cell.
"""
import re
import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_HEADER_RE = re.compile(r"^STEP t=(\d+)\s+layer=(\S+)")
ROW_RE = re.compile(
    r"^\s*(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$"
)


def parse_log(path: pathlib.Path):
    """Yields (step, layer, m, abs_err, rel_err) for every m row in every
    STEP block in one grid_layer{L}.txt file."""
    step = layer = None
    for line in path.read_text().splitlines():
        m = STEP_HEADER_RE.match(line.strip())
        if m:
            step, layer = int(m.group(1)), m.group(2)
            continue
        m = ROW_RE.match(line)
        if m and step is not None:
            m_val, _sum, _dF, abs_err, rel_err = m.groups()
            yield step, layer, int(m_val), float(abs_err), float(rel_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--m", type=int, default=1000,
                     help="Which m row to plot (must be in GRID_M_LIST).")
    args = ap.parse_args()

    log_dir = pathlib.Path(args.log_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = {}  # (layer:int, step:int) -> (abs_err, rel_err)
    for f in sorted(log_dir.glob("grid_layer*.txt")):
        for step, layer, m_val, abs_err, rel_err in parse_log(f):
            if m_val != args.m:
                continue
            cells[(int(layer), step)] = (abs_err, rel_err)

    if not cells:
        raise SystemExit(f"No cells parsed at m={args.m} from {log_dir}/grid_layer*.txt "
                          f"-- check the log format / --m value.")

    layers = sorted({l for l, _ in cells})
    steps = sorted({s for _, s in cells})
    print(f"[INFO] parsed {len(cells)} cells: {len(layers)} layers x {len(steps)} steps")

    abs_grid = np.full((len(layers), len(steps)), np.nan)
    rel_grid = np.full((len(layers), len(steps)), np.nan)
    for (l, s), (a, r) in cells.items():
        i, j = layers.index(l), steps.index(s)
        abs_grid[i, j] = a
        rel_grid[i, j] = r

    max_abs = np.nanmax(abs_grid)
    max_rel = np.nanmax(rel_grid)
    i_a, j_a = np.unravel_index(np.nanargmax(abs_grid), abs_grid.shape)
    i_r, j_r = np.unravel_index(np.nanargmax(rel_grid), rel_grid.shape)
    print(f"[SUMMARY] max abs_err={max_abs:.3e} at (layer={layers[i_a]}, t={steps[j_a]})")
    print(f"[SUMMARY] max rel_err={max_rel:.3e} at (layer={layers[i_r]}, t={steps[j_r]})")

    for grid, name, label in [
        (rel_grid, "rel_err", "relative error"),
        (abs_grid, "abs_err", "absolute error"),
    ]:
        fig, ax = plt.subplots(figsize=(0.45 * len(steps) + 2, 0.35 * len(layers) + 1.5))
        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps, fontsize=7)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
        ax.set_xlabel("denoising step $t$")
        ax.set_ylabel("layer $\\ell$")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(label)
        fig.tight_layout()
        out_path = out_dir / f"completeness_{name}_grid.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
