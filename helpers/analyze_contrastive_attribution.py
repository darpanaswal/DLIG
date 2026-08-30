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


def is_content_token(token):
    """Cleans token and checks if it's a valid semantic content word."""
    clean_tok = token.replace("\u0120", "").replace("\u2581", "").strip("_ ").lower()
    if not clean_tok or not re.search(r'[a-z]', clean_tok):
        return False
    if clean_tok in STOPWORDS:
        return False
    return True


def aggregate_by_step_layer(input_file, want_label):
    """
    Returns {step: {layer: [peak_content_scores across prompts]}} for one label.
    """
    agg = defaultdict(lambda: defaultdict(list))
    if not os.path.exists(input_file):
        print(f"[WARNING] Input file not found: {input_file}")
        return agg

    with open(input_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") != want_label:
                continue
            tokens = row.get("input_tokens", [])
            valid_indices = [i for i, tok in enumerate(tokens) if is_content_token(tok)]
            if not valid_indices:
                valid_indices = list(range(len(tokens)))
            for step_data in row.get("steps_data", []):
                step = step_data.get("step")
                for layer_str, token_scores in step_data.get("layers", {}).items():
                    layer_idx = int(layer_str)
                    content_scores = [token_scores[i] for i in valid_indices if i < len(token_scores)]
                    if content_scores:
                        agg[step][layer_idx].append(max(content_scores))
    return agg


def compute_layer_curves(agg_step):
    """
    Given agg for ONE step: {layer: [scores]}, return (layers, means, stderrs)
    with within-layer normalization.
    """
    layers = sorted(agg_step.keys())
    means, stderrs = [], []
    for layer in layers:
        raw = agg_step[layer]
        layer_max = max((abs(s) for s in raw), default=0.0)
        if layer_max == 0.0:
            means.append(0.0)
            stderrs.append(0.0)
            continue
        norm = [s / layer_max for s in raw]
        means.append(np.mean(norm))
        stderrs.append(np.std(norm) / np.sqrt(len(norm)))
    return layers, np.array(means), np.array(stderrs)


def plot_single_experiment(agg, title, output_file, cmap_name):
    """Subplots for each timestep (single class)."""
    if not agg:
        print(f"[SKIP] No data to plot for {output_file}")
        return

    steps = sorted(agg.keys())
    num_steps = len(steps)
    
    # Dynamic grid sizing: max 2 columns
    cols = min(2, num_steps)
    rows = math.ceil(num_steps / cols) if num_steps > 0 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False, sharey=True)
    axes = axes.flatten()
    cmap = plt.get_cmap(cmap_name)

    for i, step in enumerate(steps):
        ax = axes[i]
        # Use a darker shade from the colormap for visibility
        shade = 0.8 
        layers, means, stderrs = compute_layer_curves(agg[step])
        
        ax.errorbar(
            layers, means, yerr=stderrs,
            label=f"Step {step}", color=cmap(shade),
            marker='o', markersize=5, linewidth=2, capsize=3, alpha=0.9,
        )

        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.set_title(f"Timestep t={step}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Model Layer", fontsize=10)
        if i % cols == 0:
            ax.set_ylabel("Normalized DLIG", fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # Hide any unused subplots if the grid has empty slots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved {output_file}")


def plot_forced_contrastive(agg_harmful, agg_benign, title, output_file):
    """Subplots for each timestep (harmful vs benign contrastive)."""
    if not agg_harmful and not agg_benign:
        print(f"[SKIP] No data to plot for {output_file}")
        return

    steps = sorted(set(agg_harmful.keys()) | set(agg_benign.keys()))
    num_steps = len(steps)
    
    # Dynamic grid sizing: max 2 columns
    cols = min(2, num_steps)
    rows = math.ceil(num_steps / cols) if num_steps > 0 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False, sharey=True)
    axes = axes.flatten()
    
    reds = plt.get_cmap("Reds")
    blues = plt.get_cmap("Blues")

    for i, step in enumerate(steps):
        ax = axes[i]
        
        if step in agg_harmful:
            layers, means, stderrs = compute_layer_curves(agg_harmful[step])
            ax.errorbar(layers, means, yerr=stderrs, color=reds(0.8),
                        marker='o', markersize=5, linewidth=2, capsize=3,
                        alpha=0.9, label=f"Harmful")
        if step in agg_benign:
            layers, means, stderrs = compute_layer_curves(agg_benign[step])
            ax.errorbar(layers, means, yerr=stderrs, color=blues(0.8),
                        marker='s', markersize=5, linewidth=2, capsize=3,
                        alpha=0.9, label=f"Benign")

        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.set_title(f"Timestep t={step}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Model Layer", fontsize=10)
        if i % cols == 0:
            ax.set_ylabel("Normalized DLIG", fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default="diffugpt", choices=["dream", "diffugpt"],
                        help="Model family. Controls default filenames and plot titles.")
    parser.add_argument("--out_dir", type=str, default="outputs/contrast",
                        help="Directory to save the plots and read default logs from.")
    parser.add_argument("--unforced_file", type=str, default=None,
                        help="Override default unforced input file.")
    parser.add_argument("--forced_file", type=str, default=None,
                        help="Override default forced input file.")
    args = parser.parse_args()

    fam = args.family.lower()
    fam_display = "Dream" if fam == "dream" else "DiffuGPT"
    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve input files
    unforced_file = args.unforced_file or os.path.join(args.out_dir, f"{fam}_attribution_unforced.jsonl")
    forced_file = args.forced_file or os.path.join(args.out_dir, f"{fam}_attribution_forced_refusal.jsonl")

    # Resolve output files
    out_harmful = os.path.join(args.out_dir, f"plot_{fam}_harmful_unforced.png")
    out_benign = os.path.join(args.out_dir, f"plot_{fam}_benign_unforced.png")
    out_forced = os.path.join(args.out_dir, f"plot_{fam}_forced_refusal.png")

    # Exp 1: harmful unforced
    print(f"[INFO] Processing {fam_display} Harmful (unforced) from {unforced_file}...")
    harmful_unforced = aggregate_by_step_layer(unforced_file, "harmful")
    plot_single_experiment(
        harmful_unforced,
        f"[{fam_display}] Harmful Prompts — Self-Generated Target",
        out_harmful, cmap_name="Reds",
    )

    # Exp 2: benign unforced
    print(f"[INFO] Processing {fam_display} Benign (unforced) from {unforced_file}...")
    benign_unforced = aggregate_by_step_layer(unforced_file, "benign")
    plot_single_experiment(
        benign_unforced,
        f"[{fam_display}] Benign Prompts — Self-Generated Target",
        out_benign, cmap_name="Blues",
    )

    # Exp 3: forced refusal, harmful vs benign
    print(f"[INFO] Processing {fam_display} Forced refusal (both classes) from {forced_file}...")
    forced_harmful = aggregate_by_step_layer(forced_file, "harmful")
    forced_benign = aggregate_by_step_layer(forced_file, "benign")
    plot_forced_contrastive(
        forced_harmful, forced_benign,
        f"[{fam_display}] Forced Refusal Target — Harmful vs Benign",
        out_forced,
    )


if __name__ == "__main__":
    main()