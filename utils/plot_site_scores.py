#!/usr/bin/env python3
"""
Plot contrastive DLIG site scores for analysis.

Reads site_scores_S_l_t.json and produces:
1. Heatmap of S_diff across (layer, timestep) — the primary output.
2. Heatmaps of S_harm and S_benign individually.
3. Bar chart of top-K sites ranked by |S_diff|.
4. Per-layer line plots showing S_harm and S_benign over timesteps.

Usage:
    python plot_site_scores.py --input outputs/contrastive/site_scores_S_l_t.json
    python plot_site_scores.py --input outputs/contrastive/site_scores_S_l_t.json --output_dir plots/
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Any, List, Tuple
from utils.config import CONTRAST_OUTPUT, CONTRAST_PLOTS


def load_site_scores(path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    with open(path, "r") as f:
        return json.load(f)


def extract_grid(
    scores: Dict[str, Dict[str, Dict[str, Any]]],
    field: str,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Extract a 2D array [n_layers, n_steps] for a given field (e.g. 'diff_mean').

    Returns:
        grid: np.ndarray [n_layers, n_steps]
        layer_labels: sorted layer strings
        step_labels: sorted step ints
    """
    layers = sorted(scores.keys(), key=lambda x: int(x) if x.isdigit() else -1)
    all_steps = set()
    for layer_data in scores.values():
        for step_str in layer_data.keys():
            all_steps.add(int(step_str))
    steps = sorted(all_steps)

    grid = np.full((len(layers), len(steps)), np.nan)
    for i, layer in enumerate(layers):
        for j, step in enumerate(steps):
            step_str = str(step)
            if step_str in scores.get(layer, {}):
                val = scores[layer][step_str].get(field, None)
                if val is not None:
                    grid[i, j] = val

    return grid, layers, steps


def plot_heatmap(
    grid: np.ndarray,
    layer_labels: List[str],
    step_labels: List[int],
    title: str,
    cmap: str = "RdBu_r",
    center_zero: bool = True,
    output_path: str = None,
):
    """Plot a single heatmap."""
    fig, ax = plt.subplots(figsize=(max(8, len(step_labels) * 0.8), max(4, len(layer_labels) * 0.7)))

    if center_zero and not np.all(np.isnan(grid)):
        vmax = np.nanmax(np.abs(grid))
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    else:
        norm = None

    im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(step_labels)))
    ax.set_xticklabels([str(s) for s in step_labels], fontsize=9)
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels([f"Layer {l}" for l in layer_labels], fontsize=10)

    ax.set_xlabel("Diffusion Timestep", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Annotate cells with values
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.6 * np.nanmax(np.abs(grid)) else "black"
                ax.text(j, i, f"{val:.2e}", ha="center", va="center",
                        fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[SAVED] {output_path}")
    plt.close(fig)


def plot_top_sites_bar(
    scores: Dict[str, Dict[str, Dict[str, Any]]],
    top_k: int = 15,
    output_path: str = None,
):
    """Bar chart of top sites ranked by |S_diff|."""
    sites = []
    for layer, steps in scores.items():
        for step, data in steps.items():
            if "diff_mean" in data:
                sites.append({
                    "label": f"L{layer}/t{step}",
                    "diff": data["diff_mean"],
                    "abs_diff": data["diff_abs_mean"],
                    "harm": data.get("harm_mean", 0),
                    "benign": data.get("benign_mean", 0),
                })

    sites.sort(key=lambda x: x["abs_diff"], reverse=True)
    sites = sites[:top_k]

    if not sites:
        print("[WARN] No sites with diff_mean found. Skipping bar chart.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(sites) * 0.6), 5))

    labels = [s["label"] for s in sites]
    diffs = [s["diff"] for s in sites]
    colors = ["#d62728" if d > 0 else "#1f77b4" for d in diffs]

    ax.bar(range(len(labels)), diffs, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("S_diff (harm − benign)", fontsize=11)
    ax.set_title(f"Top {top_k} Sites by |S_diff|", fontsize=13, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", edgecolor="black", label="Harm > Benign"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Benign > Harm"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[SAVED] {output_path}")
    plt.close(fig)


def plot_per_layer_lines(
    scores: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: str = None,
):
    """Per-layer line plots: S_harm and S_benign over timesteps."""
    layers = sorted(scores.keys(), key=lambda x: int(x) if x.isdigit() else -1)

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4), sharey=True)
    if n_layers == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        steps_data = scores[layer]
        step_ints = sorted([int(s) for s in steps_data.keys()])

        harm_vals = [steps_data[str(s)].get("harm_mean", np.nan) for s in step_ints]
        benign_vals = [steps_data[str(s)].get("benign_mean", np.nan) for s in step_ints]

        ax.plot(step_ints, harm_vals, "o-", color="#d62728", label="Harmful", linewidth=1.5, markersize=4)
        ax.plot(step_ints, benign_vals, "s-", color="#1f77b4", label="Benign", linewidth=1.5, markersize=4)

        ax.fill_between(step_ints, harm_vals, benign_vals, alpha=0.15, color="gray")

        ax.set_xlabel("Timestep", fontsize=10)
        ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean |DLIG| (L2 norm)", fontsize=10)
    fig.suptitle("DLIG Attribution by Layer and Timestep", fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[SAVED] {output_path}")
    plt.close(fig)


def plot_harm_benign_heatmaps(
    scores: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: str,
):
    """Side-by-side heatmaps for S_harm and S_benign."""
    harm_grid, layers, steps = extract_grid(scores, "harm_mean")
    benign_grid, _, _ = extract_grid(scores, "benign_mean")

    vmax = max(np.nanmax(harm_grid) if not np.all(np.isnan(harm_grid)) else 0,
               np.nanmax(benign_grid) if not np.all(np.isnan(benign_grid)) else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(steps) * 1.4), max(4, len(layers) * 0.7)))

    for ax, grid, title in [(ax1, harm_grid, "S_harm (Harmful Prompts)"),
                             (ax2, benign_grid, "S_benign (Benign Prompts)")]:
        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([str(s) for s in steps], fontsize=9)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"Layer {l}" for l in layers], fontsize=10)
        ax.set_xlabel("Diffusion Timestep", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > 0.6 * vmax else "black"
                    ax.text(j, i, f"{val:.2e}", ha="center", va="center",
                            fontsize=7, color=text_color)

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="Mean |DLIG|")

    fig.subplots_adjust(wspace=0.25)
    path = os.path.join(output_dir, "heatmap_harm_benign.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[SAVED] {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot contrastive DLIG site scores")
    parser.add_argument("--input", type=str, default=CONTRAST_OUTPUT / "site_scores_S_l_t.json",
                        help="Path to site_scores_S_l_t.json")
    parser.add_argument("--output_dir", type=str, default=CONTRAST_PLOTS,
                        help="Directory for plot outputs (default: same dir as input)")
    parser.add_argument("--top_k", type=int, default=15,
                        help="Number of top sites in bar chart")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input), "plots")

    scores = load_site_scores(args.input)

    # Sanity check
    n_layers = len(scores)
    n_steps = max(len(v) for v in scores.values()) if scores else 0
    has_diff = any(
        "diff_mean" in step_data
        for layer_data in scores.values()
        for step_data in layer_data.values()
    )
    print(f"[INFO] Loaded scores: {n_layers} layers, {n_steps} timesteps, diff_mean present: {has_diff}")

    if not has_diff:
        print("[WARN] No diff_mean found. Check that both harmful and benign prompts were processed.")
        print("[WARN] Plotting harm/benign heatmaps only.")

    # 1. S_diff heatmap (primary output)
    if has_diff:
        diff_grid, layers, steps = extract_grid(scores, "diff_mean")
        plot_heatmap(
            diff_grid, layers, steps,
            title="S_diff = S_harm − S_benign  (Contrastive Site Scores)",
            cmap="RdBu_r",
            center_zero=True,
            output_path=os.path.join(args.output_dir, "heatmap_S_diff.png"),
        )

    # 2. Side-by-side harm & benign heatmaps
    plot_harm_benign_heatmaps(scores, args.output_dir)

    # 3. Top sites bar chart
    if has_diff:
        plot_top_sites_bar(
            scores,
            top_k=args.top_k,
            output_path=os.path.join(args.output_dir, "bar_top_sites.png"),
        )

    # 4. Per-layer line plots
    plot_per_layer_lines(
        scores,
        output_path=os.path.join(args.output_dir, "lines_per_layer.png"),
    )

    print(f"\n[OK] All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()