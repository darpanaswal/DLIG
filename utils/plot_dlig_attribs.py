"""
Script to plot DLIG attribution scores across timesteps and layers.
Adaptable for any number of layers.

Fixes:
- Time-averaged plot y-axis scaling now uses symmetric scaling around 0 for readable comparisons.
- Removed error bars / lines from time-averaged plots (no yerr).

Additions:
- Aggregate timestep curves per layer
- Key token trajectories across timesteps
- Critical timestep identification
- Layer comparison at specific timesteps
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.config import OUTPUT_DIR
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

    # ==================== NEW VISUALIZATION METHODS ====================

    def plot_aggregate_timestep_curves(self):
        """
        Plot mean|attribution| across all tokens vs timestep for each layer.
        This shows WHEN each layer is most active in processing.
        """
        if not self.data:
            self.load_data()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        layer_names = list(self.data.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))

        # Plot 1: Mean Absolute Attribution
        ax1 = axes[0]
        for layer_name, color in zip(layer_names, colors):
            layer_data = self.data[layer_name]
            timesteps = [d['step'] for d in layer_data]
            mean_abs = [np.abs(d['scores']).mean() for d in layer_data]
            ax1.plot(timesteps, mean_abs, label=layer_name, color=color, marker='o', markersize=4, linewidth=2)

        ax1.set_xlabel("Diffusion Timestep", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Mean |Attribution|", fontsize=12, fontweight='bold')
        ax1.set_title("Attribution Magnitude Over Time", fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale often helps see patterns

        # Plot 2: Max Absolute Attribution (peak signal)
        ax2 = axes[1]
        for layer_name, color in zip(layer_names, colors):
            layer_data = self.data[layer_name]
            timesteps = [d['step'] for d in layer_data]
            max_abs = [np.abs(d['scores']).max() for d in layer_data]
            ax2.plot(timesteps, max_abs, label=layer_name, color=color, marker='s', markersize=4, linewidth=2)

        ax2.set_xlabel("Diffusion Timestep", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Max |Attribution|", fontsize=12, fontweight='bold')
        ax2.set_title("Peak Attribution Signal Over Time", fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "aggregate_timestep_curves.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved aggregate timestep curves to: {output_path}")
        plt.close()

    def plot_key_token_trajectories(self, key_tokens=None):
        """
        Track specific safety-relevant tokens across timesteps for each layer.
        
        Args:
            key_tokens: List of token strings to track. If None, uses default safety tokens.
        """
        if not self.data:
            self.load_data()

        if key_tokens is None:
            key_tokens = ["jew", "hate", "promote", "speech", "discrimination", "against"]

        layer_names = list(self.data.keys())
        num_layers = len(layer_names)

        fig, axes = plt.subplots(num_layers, 1, figsize=(14, 4 * num_layers), squeeze=False)
        axes = axes.flatten()

        colors = plt.cm.tab10(np.linspace(0, 1, len(key_tokens)))

        for layer_idx, layer_name in enumerate(layer_names):
            ax = axes[layer_idx]
            layer_data = self.data[layer_name]
            
            # Get token list from first timestep
            tokens = layer_data[0]["tokens"]
            timesteps = [d['step'] for d in layer_data]

            # Find indices of key tokens (handle variations with/without space prefix)
            token_indices = {}
            for kt in key_tokens:
                for idx, t in enumerate(tokens):
                    t_clean = t.strip().lower()
                    if t_clean == kt.lower() or t_clean == kt.lower().lstrip():
                        token_indices[kt] = idx
                        break

            # Plot trajectory for each found token
            for (token_name, token_idx), color in zip(token_indices.items(), colors):
                values = [d['scores'][token_idx] for d in layer_data]
                ax.plot(timesteps, values, label=f'"{token_name}"', color=color, 
                       marker='o', markersize=5, linewidth=2)

            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_xlabel("Diffusion Timestep", fontsize=11, fontweight='bold')
            ax.set_ylabel("Attribution Score", fontsize=11, fontweight='bold')
            ax.set_title(f"{layer_name}: Key Token Attribution Trajectories", fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)

            # Add shaded regions for different phases
            max_step = max(timesteps)
            ax.axvspan(0, max_step * 0.3, alpha=0.1, color='blue', label='Early (semantic)')
            ax.axvspan(max_step * 0.3, max_step * 0.7, alpha=0.1, color='green', label='Mid (reasoning)')
            ax.axvspan(max_step * 0.7, max_step, alpha=0.1, color='orange', label='Late (output)')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "key_token_trajectories.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved key token trajectories to: {output_path}")
        plt.close()

    def plot_critical_timesteps(self, top_k=10):
        """
        Identify and visualize timesteps with largest attribution changes.
        These are potential "decision points" in the diffusion process.
        """
        if not self.data:
            self.load_data()

        layer_names = list(self.data.keys())
        
        fig, axes = plt.subplots(len(layer_names), 1, figsize=(14, 4 * len(layer_names)), squeeze=False)
        axes = axes.flatten()

        all_critical = {}

        for layer_idx, layer_name in enumerate(layer_names):
            ax = axes[layer_idx]
            layer_data = self.data[layer_name]
            
            timesteps = [d['step'] for d in layer_data]
            
            # Compute change magnitude between consecutive timesteps
            changes = []
            for i in range(1, len(layer_data)):
                prev_scores = layer_data[i-1]['scores']
                curr_scores = layer_data[i]['scores']
                delta = np.abs(curr_scores - prev_scores).sum()
                changes.append({
                    'step': layer_data[i]['step'],
                    'prev_step': layer_data[i-1]['step'],
                    'delta': delta
                })

            # Sort by delta to find critical steps
            changes_sorted = sorted(changes, key=lambda x: x['delta'], reverse=True)
            critical_steps = [c['step'] for c in changes_sorted[:top_k]]
            all_critical[layer_name] = changes_sorted[:top_k]

            # Plot delta over time
            steps_for_plot = [c['step'] for c in changes]
            deltas_for_plot = [c['delta'] for c in changes]

            ax.bar(steps_for_plot, deltas_for_plot, color='steelblue', edgecolor='black', alpha=0.7)
            
            # Highlight critical steps
            for c in changes_sorted[:top_k]:
                idx = steps_for_plot.index(c['step'])
                ax.bar(c['step'], c['delta'], color='red', edgecolor='black', alpha=0.9)

            ax.set_xlabel("Diffusion Timestep", fontsize=11, fontweight='bold')
            ax.set_ylabel("Σ|Δ Attribution|", fontsize=11, fontweight='bold')
            ax.set_title(f"{layer_name}: Attribution Change Magnitude (Red = Top {top_k} Critical)", 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "critical_timesteps.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved critical timesteps plot to: {output_path}")
        plt.close()

        # Also save critical steps as JSON
        critical_json_path = os.path.join(self.output_dir, "critical_timesteps.json")
        with open(critical_json_path, 'w') as f:
            # Convert to serializable format
            serializable = {k: [{'step': c['step'], 'delta': float(c['delta'])} for c in v] 
                          for k, v in all_critical.items()}
            json.dump(serializable, f, indent=2)
        print(f"Saved critical timesteps data to: {critical_json_path}")

        return all_critical

    def plot_layer_comparison_at_timesteps(self, timesteps_to_compare=None):
        """
        Compare attribution patterns across layers at specific timesteps.
        Useful for seeing how information flows through the network at key moments.
        """
        if not self.data:
            self.load_data()

        layer_names = list(self.data.keys())
        
        # Default to early, mid, late timesteps
        if timesteps_to_compare is None:
            sample_layer = self.data[layer_names[0]]
            all_steps = [d['step'] for d in sample_layer]
            max_step = max(all_steps)
            timesteps_to_compare = [
                min(all_steps),  # First
                all_steps[len(all_steps) // 4],  # Early
                all_steps[len(all_steps) // 2],  # Mid
                all_steps[3 * len(all_steps) // 4],  # Late
                max(all_steps),  # Final
            ]

        num_timesteps = len(timesteps_to_compare)
        fig, axes = plt.subplots(num_timesteps, 1, figsize=(14, 4 * num_timesteps), squeeze=False)
        axes = axes.flatten()

        for t_idx, target_step in enumerate(timesteps_to_compare):
            ax = axes[t_idx]
            
            # Get tokens from first layer
            tokens = self.data[layer_names[0]][0]['tokens']
            x_pos = np.arange(len(tokens))
            width = 0.8 / len(layer_names)

            for l_idx, layer_name in enumerate(layer_names):
                layer_data = self.data[layer_name]
                
                # Find the step closest to target
                step_data = None
                for d in layer_data:
                    if d['step'] == target_step:
                        step_data = d
                        break
                
                if step_data is None:
                    # Find closest
                    step_data = min(layer_data, key=lambda d: abs(d['step'] - target_step))

                scores = step_data['scores']
                offset = (l_idx - len(layer_names) / 2 + 0.5) * width
                ax.bar(x_pos + offset, scores, width, label=layer_name, alpha=0.8)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel("Attribution Score", fontsize=11, fontweight='bold')
            ax.set_title(f"Layer Comparison at Timestep {target_step}", fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8, ncol=len(layer_names))
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "layer_comparison_timesteps.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved layer comparison plot to: {output_path}")
        plt.close()

    def plot_token_attribution_summary(self, key_tokens=None):
        """
        Create a summary heatmap: layers (rows) × key tokens (cols), showing time-averaged attribution.
        This gives a quick overview of which layers attend to which safety tokens.
        """
        if not self.data:
            self.load_data()

        if key_tokens is None:
            key_tokens = ["How", "to", "promote", "hate", "speech", "and", "discrimination", "against", "jew", "s", "?"]

        layer_names = list(self.data.keys())
        
        # Build matrix: layers × tokens
        matrix = np.zeros((len(layer_names), len(key_tokens)))
        
        for l_idx, layer_name in enumerate(layer_names):
            layer_data = self.data[layer_name]
            tokens = layer_data[0]['tokens']
            
            # Average across all timesteps
            all_scores = np.stack([d['scores'] for d in layer_data], axis=0)
            avg_scores = np.mean(all_scores, axis=0)
            
            for t_idx, kt in enumerate(key_tokens):
                # Find token index
                found_idx = None
                for idx, t in enumerate(tokens):
                    t_clean = t.strip().lower()
                    if t_clean == kt.lower() or t_clean == kt.lower().lstrip():
                        found_idx = idx
                        break
                
                if found_idx is not None:
                    matrix[l_idx, t_idx] = avg_scores[found_idx]

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use diverging colormap centered at 0
        max_abs = np.abs(matrix).max()
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-max_abs, vmax=max_abs)

        ax.set_xticks(range(len(key_tokens)))
        ax.set_xticklabels(key_tokens, rotation=45, ha='right', fontsize=11)
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=11)

        ax.set_xlabel("Token", fontsize=12, fontweight='bold')
        ax.set_ylabel("Layer", fontsize=12, fontweight='bold')
        ax.set_title("Time-Averaged Attribution: Layers × Tokens", fontsize=14, fontweight='bold')

        # Add value annotations
        for i in range(len(layer_names)):
            for j in range(len(key_tokens)):
                val = matrix[i, j]
                color = 'white' if abs(val) > max_abs * 0.5 else 'black'
                text = f"{val:.2e}" if abs(val) < 0.001 else f"{val:.4f}"
                ax.text(j, i, text, ha='center', va='center', fontsize=7, color=color)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attribution Score", fontsize=11)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "token_attribution_summary.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved token attribution summary to: {output_path}")
        plt.close()

    def generate_all_plots(self, max_tokens_display=15, key_tokens=None):
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

        print("\n4. Creating aggregate timestep curves...")
        self.plot_aggregate_timestep_curves()

        print("\n5. Creating key token trajectories...")
        self.plot_key_token_trajectories(key_tokens=key_tokens)

        print("\n6. Identifying critical timesteps...")
        self.plot_critical_timesteps(top_k=10)

        print("\n7. Creating layer comparison at key timesteps...")
        self.plot_layer_comparison_at_timesteps()

        print("\n8. Creating token attribution summary heatmap...")
        self.plot_token_attribution_summary(key_tokens=key_tokens)

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
  
  # Specify key tokens to track
  python plot_attributions.py --steps 32 --layers 0 7 14 \\
                               --key-tokens jew hate promote speech
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
    parser.add_argument(
        "--key-tokens",
        nargs="+",
        default=None,
        help="Key tokens to track in trajectory plots (default: safety-relevant tokens)",
    )

    args = parser.parse_args()

    output_dir = OUTPUT_DIR / f"{args.steps}_steps"
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

    plots_output_dir = f"{output_dir}/plots"

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
    if args.key_tokens:
        print(f"Key tokens to track: {args.key_tokens}")
    print("=" * 60)

    plotter = AttributionPlotter(
        csv_files=csv_files,
        output_dir=plots_output_dir,
        figsize_per_step=tuple(args.figsize),
    )

    plotter.generate_all_plots(max_tokens_display=args.max_tokens, key_tokens=args.key_tokens)


if __name__ == "__main__":
    main()