"""
Script to plot DLIG attribution scores across timesteps and layers.
Adaptable for any number of layers.

Fixes:
- Time-averaged plot y-axis scaling now uses symmetric scaling around 0 for readable comparisons.
- Removed error bars / lines from time-averaged plots (no yerr).
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Increase CSV field size limit to handle large token_scores JSON
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)


class AttributionPlotter:
    def __init__(self, csv_files, output_dir="./plots", figsize_per_step=(3, 2)):
        """
        Initialize the attribution plotter.

        Args:
            csv_files: List of CSV file paths or dict mapping layer names to file paths
            output_dir: Directory to save plots
            figsize_per_step: Size of each subplot (width, height)
        """
        self.csv_files = (
            csv_files
            if isinstance(csv_files, dict)
            else {f"Layer {i}": f for i, f in enumerate(csv_files)}
        )
        self.output_dir = output_dir
        self.figsize_per_step = figsize_per_step
        self.data = {}

        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        """Load attribution data from CSV files."""
        print("Loading attribution data...")

        for layer_name, csv_file in self.csv_files.items():
            if not os.path.exists(csv_file):
                print(f"Warning: File not found: {csv_file}")
                continue

            layer_data = []
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    step = int(row["step"])
                    token_scores = json.loads(row["token_scores"])

                    # Convert to lists for indexed access
                    all_tokens = list(token_scores.keys())
                    all_scores = [
                        score_info["flattened_score"] for score_info in token_scores.values()
                    ]

                    # Find the LAST occurrence of 'user' role marker and FIRST 'assistant' after it
                    user_content_start = -1
                    user_content_end = len(all_tokens)

                    for i in range(len(all_tokens) - 1, -1, -1):
                        token_clean = all_tokens[i].replace("Ġ", "").strip().lower()
                        if token_clean == "user":
                            user_content_start = i
                            break

                    if user_content_start >= 0:
                        for i in range(user_content_start + 1, len(all_tokens)):
                            token_clean = all_tokens[i].replace("Ġ", "").strip().lower()
                            if token_clean == "assistant":
                                user_content_end = i
                                break

                    tokens = []
                    scores = []

                    if user_content_start >= 0:
                        for i in range(user_content_start + 1, user_content_end):
                            token = all_tokens[i]
                            clean_token = token.replace("Ġ", " ")
                            tokens.append(clean_token)
                            scores.append(all_scores[i])
                    else:
                        print(f"Warning: No 'user' marker found in step {step}, using all tokens")
                        for token, score in zip(all_tokens, all_scores):
                            clean_token = token.replace("Ġ", " ")
                            tokens.append(clean_token)
                            scores.append(score)

                    if tokens:
                        layer_data.append(
                            {"step": step, "tokens": tokens, "scores": np.array(scores, dtype=float)}
                        )

            if layer_data:
                self.data[layer_name] = sorted(layer_data, key=lambda x: x["step"])
                print(f"Loaded {len(layer_data)} timesteps for {layer_name}")
                print(f"  - Tokens per step: {len(layer_data[0]['tokens'])}")
                print(f"  - First few tokens: {layer_data[0]['tokens'][:5]}")
                print(f"  - Last few tokens: {layer_data[0]['tokens'][-3:]}")
            else:
                print(f"Warning: No data loaded for {layer_name}")

        if not self.data:
            raise ValueError("No data loaded from any CSV files!")

        return self.data

    @staticmethod
    def _compute_symmetric_ylim(values: np.ndarray, margin_frac: float = 0.12, min_half_range: float = 1e-6):
        """
        Compute symmetric y-limits around 0 for readability.
        """
        values = np.asarray(values, dtype=float)
        if values.size == 0 or not np.isfinite(values).any():
            return (-min_half_range, min_half_range)

        max_abs = float(np.nanmax(np.abs(values[np.isfinite(values)])))
        half_range = max(max_abs * (1.0 + margin_frac), min_half_range)
        return (-half_range, half_range)

    def plot_timestep_attributions(self, max_tokens_display=15):
        """
        Create a grid plot showing attributions at each timestep for all layers.
        Each individual subplot uses its own color scale and y-axis range.
        """
        if not self.data:
            self.load_data()

        num_layers = len(self.data)
        layer_names = list(self.data.keys())

        max_steps = max(len(self.data[layer]) for layer in layer_names)

        fig_width = self.figsize_per_step[0] * max_steps
        fig_height = self.figsize_per_step[1] * num_layers

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(num_layers, max_steps, figure=fig, hspace=0.4, wspace=0.3)

        for layer_idx, layer_name in enumerate(layer_names):
            layer_data = self.data[layer_name]

            for step_idx, step_data in enumerate(layer_data):
                ax = fig.add_subplot(gs[layer_idx, step_idx])

                tokens = step_data["tokens"][:max_tokens_display]
                scores = step_data["scores"][:max_tokens_display]
                step = step_data["step"]

                vmin = np.min(scores)
                vmax = np.max(scores)

                if vmax - vmin > 1e-10:
                    colors = plt.cm.RdYlGn((scores - vmin) / (vmax - vmin))
                else:
                    colors = ["gray"] * len(scores)

                ax.bar(
                    range(len(scores)),
                    scores,
                    color=colors,
                    edgecolor="black",
                    linewidth=0.5,
                )

                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=6)
                ax.tick_params(axis="y", labelsize=6)
                ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

                if layer_idx == 0:
                    ax.set_title(f"Step {step}", fontsize=8, fontweight="bold")

                if step_idx == 0:
                    ax.set_ylabel(f"{layer_name}\nScore", fontsize=7)

                y_margin = (vmax - vmin) * 0.1 if vmax - vmin > 1e-10 else 0.0001
                ax.set_ylim(vmin - y_margin, vmax + y_margin)

                ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
                ax.set_axisbelow(True)

        fig.suptitle(
            "Token-wise Attribution Scores Across Timesteps",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )

        output_path = os.path.join(self.output_dir, "timestep_attributions.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved timestep attribution plot to: {output_path}")
        plt.close()

    def plot_averaged_attributions(self, max_tokens_display=20):
        """
        Create a plot showing averaged attributions across all timesteps for each layer.

        Changes:
        - Removed error bars/lines (no std overlays).
        - Y-axis scaling is symmetric around 0 based on layer max absolute avg score.
        """
        if not self.data:
            self.load_data()

        num_layers = len(self.data)
        layer_names = list(self.data.keys())

        averaged_data = {}
        for layer_name in layer_names:
            layer_data = self.data[layer_name]

            tokens = layer_data[0]["tokens"]
            all_scores = np.stack([step_data["scores"] for step_data in layer_data], axis=0)
            avg_scores = np.mean(all_scores, axis=0)

            averaged_data[layer_name] = {"tokens": tokens, "avg_scores": avg_scores}

        fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4 * num_layers), squeeze=False)
        axes = axes.flatten()

        for idx, layer_name in enumerate(layer_names):
            ax = axes[idx]
            data = averaged_data[layer_name]

            tokens = data["tokens"][:max_tokens_display]
            avg_scores = data["avg_scores"][:max_tokens_display]

            layer_vmin = float(np.min(avg_scores))
            layer_vmax = float(np.max(avg_scores))

            x_pos = np.arange(len(tokens))

            if layer_vmax - layer_vmin > 1e-10:
                colors = plt.cm.RdYlGn((avg_scores - layer_vmin) / (layer_vmax - layer_vmin))
            else:
                colors = ["gray"] * len(avg_scores)

            bars = ax.bar(
                x_pos,
                avg_scores,
                color=colors,
                edgecolor="black",
                linewidth=1,
                alpha=0.85,
            )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)
            ax.set_ylabel("Average Attribution Score", fontsize=11, fontweight="bold")
            ax.set_title(
                f"{layer_name} - Time-Averaged Attributions",
                fontsize=12,
                fontweight="bold",
                pad=10,
            )
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
            ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)

            # Symmetric scaling around 0 (fix)
            ylow, yhigh = self._compute_symmetric_ylim(avg_scores, margin_frac=0.12, min_half_range=1e-6)
            ax.set_ylim(ylow, yhigh)

            # Value labels (kept)
            for bar, score in zip(bars, avg_scores):
                height = bar.get_height()
                if abs(score) < 0.001:
                    label = f"{score:.2e}"
                else:
                    label = f"{score:.4f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    label,
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=7,
                )

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "averaged_attributions.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved averaged attribution plot to: {output_path}")
        plt.close()

    def plot_attribution_heatmap(self):
        """Create a heatmap showing attribution evolution over timesteps. Each layer uses its own color scale."""
        if not self.data:
            self.load_data()

        num_layers = len(self.data)
        layer_names = list(self.data.keys())

        fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 8), squeeze=False)
        axes = axes.flatten()

        for idx, layer_name in enumerate(layer_names):
            ax = axes[idx]
            layer_data = self.data[layer_name]

            tokens = layer_data[0]["tokens"]
            score_matrix = np.array([step_data["scores"] for step_data in layer_data], dtype=float)
            timesteps = [step_data["step"] for step_data in layer_data]

            vmin = float(np.min(score_matrix))
            vmax = float(np.max(score_matrix))

            im = ax.imshow(
                score_matrix.T,
                aspect="auto",
                cmap="RdYlGn",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=8)
            ax.set_xticks(range(len(timesteps)))
            ax.set_xticklabels(timesteps, fontsize=8)

            ax.set_xlabel("Diffusion Timestep", fontsize=10, fontweight="bold")
            ax.set_ylabel("Tokens", fontsize=10, fontweight="bold")
            ax.set_title(f"{layer_name}\nAttribution Heatmap", fontsize=11, fontweight="bold")

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Attribution Score", fontsize=9)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "attribution_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved attribution heatmap to: {output_path}")
        plt.close()

    def generate_all_plots(self, max_tokens_display=15):
        """Generate all visualization types."""
        print("\nGenerating all plots...")
        print("=" * 60)

        self.load_data()

        print("\n1. Creating timestep attribution plot...")
        self.plot_timestep_attributions(max_tokens_display=max_tokens_display)

        print("\n2. Creating averaged attribution plot...")
        self.plot_averaged_attributions(max_tokens_display=max_tokens_display)

        print("\n3. Creating attribution heatmap...")
        self.plot_attribution_heatmap()

        print("\n" + "=" * 60)
        print("All plots generated successfully!")
        print(f"Plots saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot DLIG attribution scores from CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single layer
  python plot_attributions.py --steps 15 --layers 26

  # Multiple layers
  python plot_attributions.py --steps 15 --layers 0 7 26

  # Multiple layers with custom names
  python plot_attributions.py --steps 15 --layers 0 7 26 \\
                               --layer-names "Embed" "Layer 0" "Layer 7"
        """,
    )

    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Number of generation steps (X in outputs/X_steps/)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        required=True,
        help="Layer numbers to plot (Y in layersY_X.csv)",
    )
    parser.add_argument(
        "--layer-names",
        nargs="+",
        default=None,
        help="Custom names for layers (must match number of layers)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=15,
        help="Maximum number of tokens to display (default: 15)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[3, 2],
        help="Size of each subplot (width height) (default: 3 2)",
    )

    args = parser.parse_args()

    output_dir = f"outputs/{args.steps}_steps"
    csv_files = {}

    for layer in args.layers:
        csv_path = os.path.join(output_dir, f"layers{layer}_{args.steps}.csv")

        if not os.path.exists(csv_path):
            print(f"Warning: File not found: {csv_path}")
            continue

        if args.layer_names:
            if len(csv_files) < len(args.layer_names):
                layer_name = args.layer_names[len(csv_files)]
            else:
                layer_name = f"Layer {layer}"
        else:
            layer_name = f"Layer {layer}"

        csv_files[layer_name] = csv_path

    if args.layer_names and len(args.layer_names) != len(csv_files):
        print(
            f"Warning: Number of layer names ({len(args.layer_names)}) doesn't match "
            f"number of found CSV files ({len(csv_files)})"
        )

    if not csv_files:
        raise ValueError(
            f"No CSV files found in {output_dir}/. "
            f"Expected files like: layers{args.layers[0]}_{args.steps}.csv"
        )

    plots_output_dir = "outputs/plots"

    print("=" * 60)
    print("DLIG Attribution Plotter")
    print("=" * 60)
    print(f"\nGeneration steps: {args.steps}")
    print(f"Input directory: {output_dir}/")
    print(f"\nLayers to plot:")
    for layer_name, csv_path in csv_files.items():
        print(f"  - {layer_name}: {csv_path}")
    print(f"\nOutput directory: {plots_output_dir}")
    print(f"Max tokens to display: {args.max_tokens}")
    print("=" * 60)

    plotter = AttributionPlotter(
        csv_files=csv_files,
        output_dir=plots_output_dir,
        figsize_per_step=tuple(args.figsize),
    )

    plotter.generate_all_plots(max_tokens_display=args.max_tokens)


if __name__ == "__main__":
    main()