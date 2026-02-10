# experiments/visualize_contrastive_results.py
"""
Visualization utilities for contrastive ΔDLIG outputs.

Inputs (produced by experiments/contrastive_runner.py):
- site_scores_S_l_t.json / .pt
- per_pair_delta_dlig.json / .pt

Outputs:
- Heatmaps of S(l,t) for delta_mean, harm_mean, delta_std, etc.
- Per-layer curves S(t)
- Top-K (and optionally bottom-K) sites table
- Optional per-pair ΔDLIG curves for selected pair_ids/layers

No seaborn. Matplotlib only.
"""

import os
import csv
import json
import torch
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# New type: layer -> step -> {metric: value}
SiteScores = Dict[str, Dict[str, Dict[str, float]]]
# Legacy type for backward compatibility
SiteScoresLegacy = Dict[str, Dict[str, float]]
PerPair = List[Dict[str, Any]]


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pt(path: str) -> Any:
    return torch.load(path, map_location="cpu")


def _is_legacy_format(obj: Dict) -> bool:
    """Check if site_scores is in legacy format (layer -> step -> float)."""
    for layer, step_map in obj.items():
        if isinstance(step_map, dict):
            for step, val in step_map.items():
                # If value is a dict, it's new format; if float/int, it's legacy
                return not isinstance(val, dict)
    return True


def load_site_scores(path: str) -> Tuple[SiteScores, List[str]]:
    """
    Load site scores and return (scores_dict, available_metrics).
    
    Handles both legacy format (layer -> step -> float) and 
    new format (layer -> step -> {metric: float}).
    """
    if path.endswith(".json"):
        obj = _load_json(path)
    elif path.endswith(".pt"):
        obj = _load_pt(path)
    else:
        raise ValueError(f"Unsupported file type for site_scores: {path}")

    if not isinstance(obj, dict):
        raise ValueError("site_scores must be a dict")

    # Detect format
    if _is_legacy_format(obj):
        # Convert legacy to new format
        out: SiteScores = {}
        for layer, step_map in obj.items():
            out[str(layer)] = {}
            for step, val in step_map.items():
                out[str(layer)][str(step)] = {"mean": float(val)}
        return out, ["mean"]
    else:
        # New format
        out: SiteScores = {}
        metrics_found: set = set()
        for layer, step_map in obj.items():
            out[str(layer)] = {}
            for step, metric_dict in step_map.items():
                if isinstance(metric_dict, dict):
                    out[str(layer)][str(step)] = {str(k): float(v) for k, v in metric_dict.items()}
                    metrics_found.update(metric_dict.keys())
                else:
                    # Fallback for mixed format
                    out[str(layer)][str(step)] = {"mean": float(metric_dict)}
                    metrics_found.add("mean")
        return out, sorted(list(metrics_found))


def load_per_pair(path: str) -> PerPair:
    if path.endswith(".json"):
        obj = _load_json(path)
    elif path.endswith(".pt"):
        obj = _load_pt(path)
    else:
        raise ValueError(f"Unsupported file type for per_pair: {path}")

    if not isinstance(obj, list):
        raise ValueError("per_pair must be a list of pair records")
    return obj


def _sorted_layers(site_scores: SiteScores, layers: Optional[Sequence[str]] = None) -> List[str]:
    if layers is not None and len(layers) > 0:
        layers_list = [str(x) for x in layers]
        missing = [l for l in layers_list if l not in site_scores]
        if missing:
            raise ValueError(f"Requested layers not found in site_scores: {missing}")
        return layers_list

    def _key(s: str):
        return (0, int(s)) if s.isdigit() else (1, s)

    return sorted(list(site_scores.keys()), key=_key)


def _sorted_steps(site_scores: SiteScores, layers_sorted: Sequence[str]) -> List[str]:
    steps = set()
    for l in layers_sorted:
        for s in site_scores[l].keys():
            steps.add(str(s))

    def _skey(x: str):
        return int(x) if str(x).isdigit() else x

    return sorted(list(steps), key=_skey)


def site_scores_to_matrix(
    site_scores: SiteScores,
    layers_sorted: Sequence[str],
    steps_sorted: Sequence[str],
    metric: str = "mean",
) -> torch.Tensor:
    """Extract a specific metric into a matrix."""
    mat = torch.zeros((len(layers_sorted), len(steps_sorted)), dtype=torch.float32)
    for i, l in enumerate(layers_sorted):
        smap = site_scores.get(l, {})
        for j, s in enumerate(steps_sorted):
            if s in smap:
                metric_dict = smap[s]
                if isinstance(metric_dict, dict) and metric in metric_dict:
                    mat[i, j] = float(metric_dict[metric])
                elif isinstance(metric_dict, (int, float)):
                    mat[i, j] = float(metric_dict)
                else:
                    mat[i, j] = float("nan")
            else:
                mat[i, j] = float("nan")
    return mat


def save_heatmap(
    mat: torch.Tensor,
    layers_sorted: Sequence[str],
    steps_sorted: Sequence[str],
    out_png: str,
    out_pdf: Optional[str] = None,
    title: str = "S(l,t)",
    cmap: str = "viridis",
) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig = plt.figure(figsize=(max(8, len(steps_sorted) * 0.35), max(4, len(layers_sorted) * 0.35)))
    ax = fig.add_subplot(111)

    im = ax.imshow(mat.numpy(), aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel("diffusion step t")
    ax.set_ylabel("layer l")

    ax.set_xticks(list(range(len(steps_sorted))))
    ax.set_xticklabels([str(s) for s in steps_sorted], rotation=90)
    ax.set_yticks(list(range(len(layers_sorted))))
    ax.set_yticklabels([str(l) for l in layers_sorted])

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def save_comparison_heatmaps(
    site_scores: SiteScores,
    layers_sorted: Sequence[str],
    steps_sorted: Sequence[str],
    metrics: Sequence[str],
    out_dir: str,
    title_prefix: str = "",
) -> None:
    """Generate side-by-side heatmaps for multiple metrics."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Define colormaps for different metric types
    cmap_map = {
        "delta_mean": "Reds",
        "harm_mean": "Oranges",
        "delta_std": "Blues",
        "harm_std": "Purples",
        "delta_max": "RdPu",
        "harm_max": "YlOrRd",
        "mean": "viridis",  # Legacy
    }
    
    # Individual heatmaps
    for metric in metrics:
        mat = site_scores_to_matrix(site_scores, layers_sorted, steps_sorted, metric)
        cmap = cmap_map.get(metric, "viridis")
        
        save_heatmap(
            mat=mat,
            layers_sorted=layers_sorted,
            steps_sorted=steps_sorted,
            out_png=os.path.join(out_dir, f"heatmap_{metric}.png"),
            out_pdf=os.path.join(out_dir, f"heatmap_{metric}.pdf"),
            title=f"{title_prefix}S(l,t) - {metric}",
            cmap=cmap,
        )
    
    # Combined comparison figure (if we have both delta and harm)
    has_delta = any("delta" in m for m in metrics)
    has_harm = any("harm" in m for m in metrics)
    
    if has_delta and has_harm:
        fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(layers_sorted) * 0.35)))
        
        for ax, (metric, cmap, title) in zip(axes, [
            ("delta_mean", "Reds", "ΔDLIG (Contrastive)"),
            ("harm_mean", "Oranges", "||DLIG_harm|| (Absolute)"),
        ]):
            if metric not in metrics:
                continue
            mat = site_scores_to_matrix(site_scores, layers_sorted, steps_sorted, metric)
            im = ax.imshow(mat.numpy(), aspect="auto", interpolation="nearest", cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel("diffusion step t")
            ax.set_ylabel("layer l")
            ax.set_xticks(list(range(len(steps_sorted))))
            ax.set_xticklabels([str(s) for s in steps_sorted], rotation=90, fontsize=7)
            ax.set_yticks(list(range(len(layers_sorted))))
            ax.set_yticklabels([str(l) for l in layers_sorted])
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        
        fig.suptitle(f"{title_prefix}Contrastive vs Absolute Signal Comparison")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "comparison_delta_vs_harm.png"), dpi=200)
        fig.savefig(os.path.join(out_dir, "comparison_delta_vs_harm.pdf"))
        plt.close(fig)


def save_layer_curves(
    mat: torch.Tensor,
    layers_sorted: Sequence[str],
    steps_sorted: Sequence[str],
    out_png: str,
    out_pdf: Optional[str] = None,
    title: str = "Per-layer S(t)",
) -> None:
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    xs = list(range(len(steps_sorted)))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    for i, l in enumerate(layers_sorted):
        ys = mat[i].numpy()
        ax.plot(xs, ys, label=str(l), marker='o', markersize=3)

    ax.set_title(title)
    ax.set_xlabel("diffusion step t")
    ax.set_ylabel("S(l,t)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in steps_sorted], rotation=90)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def save_multi_metric_layer_curves(
    site_scores: SiteScores,
    layers_sorted: Sequence[str],
    steps_sorted: Sequence[str],
    metrics: Sequence[str],
    out_dir: str,
) -> None:
    """Generate per-layer curves for each metric."""
    os.makedirs(out_dir, exist_ok=True)
    
    for metric in metrics:
        mat = site_scores_to_matrix(site_scores, layers_sorted, steps_sorted, metric)
        save_layer_curves(
            mat=mat,
            layers_sorted=layers_sorted,
            steps_sorted=steps_sorted,
            out_png=os.path.join(out_dir, f"curves_{metric}.png"),
            out_pdf=os.path.join(out_dir, f"curves_{metric}.pdf"),
            title=f"Per-layer S(t) - {metric}",
        )
    
    # Overlay plot: delta_mean vs harm_mean for each layer
    if "delta_mean" in metrics and "harm_mean" in metrics:
        mat_delta = site_scores_to_matrix(site_scores, layers_sorted, steps_sorted, "delta_mean")
        mat_harm = site_scores_to_matrix(site_scores, layers_sorted, steps_sorted, "harm_mean")
        
        xs = list(range(len(steps_sorted)))
        
        # One figure per layer showing both metrics
        for i, layer in enumerate(layers_sorted):
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(xs, mat_delta[i].numpy(), label="ΔDLIG (contrastive)", color="red", marker='o', markersize=3)
            ax.plot(xs, mat_harm[i].numpy(), label="||DLIG_harm|| (absolute)", color="orange", marker='s', markersize=3)
            
            ax.set_title(f"Layer {layer}: Contrastive vs Absolute Signal")
            ax.set_xlabel("diffusion step t")
            ax.set_ylabel("Score")
            ax.set_xticks(xs)
            ax.set_xticklabels([str(s) for s in steps_sorted], rotation=90)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"overlay_layer_{layer}.png"), dpi=200)
            fig.savefig(os.path.join(out_dir, f"overlay_layer_{layer}.pdf"))
            plt.close(fig)


def save_std_bands_plot(
    site_scores: SiteScores,
    layers_sorted: Sequence[str],
    steps_sorted: Sequence[str],
    out_dir: str,
    metric_base: str = "delta",
) -> None:
    """Plot mean ± std bands for a metric."""
    mean_key = f"{metric_base}_mean"
    std_key = f"{metric_base}_std"
    
    # Check if both keys exist
    sample_layer = layers_sorted[0]
    sample_step = steps_sorted[0]
    sample_metrics = site_scores.get(sample_layer, {}).get(sample_step, {})
    
    if mean_key not in sample_metrics or std_key not in sample_metrics:
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    mat_mean = site_scores_to_matrix(site_scores, layers_sorted, steps_sorted, mean_key)
    mat_std = site_scores_to_matrix(site_scores, layers_sorted, steps_sorted, std_key)
    
    xs = list(range(len(steps_sorted)))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(range(len(layers_sorted)))
    
    for i, (layer, color) in enumerate(zip(layers_sorted, colors)):
        mean_vals = mat_mean[i].numpy()
        std_vals = mat_std[i].numpy()
        
        ax.plot(xs, mean_vals, label=f"Layer {layer}", color=color, marker='o', markersize=3)
        ax.fill_between(xs, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2)
    
    ax.set_title(f"{metric_base.upper()} Mean ± Std Across Layers")
    ax.set_xlabel("diffusion step t")
    ax.set_ylabel(f"{mean_key}")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in steps_sorted], rotation=90)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"std_bands_{metric_base}.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, f"std_bands_{metric_base}.pdf"))
    plt.close(fig)


def topk_sites(
    site_scores: SiteScores,
    top_k: int,
    metric: str = "delta_mean",
    layers: Optional[Sequence[str]] = None,
    steps: Optional[Sequence[str]] = None,
    largest: bool = True,
) -> List[Dict[str, Union[str, int, float]]]:
    """Get top-K sites by a specific metric."""
    layers_sorted = _sorted_layers(site_scores, layers)
    steps_sorted = _sorted_steps(site_scores, layers_sorted)

    if steps is not None and len(steps) > 0:
        steps_sorted = [str(s) for s in steps]

    items: List[Tuple[float, str, str, Dict[str, float]]] = []
    for l in layers_sorted:
        for s in steps_sorted:
            metric_dict = site_scores.get(l, {}).get(s, {})
            if isinstance(metric_dict, dict) and metric in metric_dict:
                val = float(metric_dict[metric])
                items.append((val, str(l), str(s), metric_dict))
            elif isinstance(metric_dict, (int, float)) and metric == "mean":
                # Legacy format
                items.append((float(metric_dict), str(l), str(s), {"mean": float(metric_dict)}))

    if not items:
        return []

    items.sort(key=lambda x: x[0], reverse=bool(largest))
    items = items[: max(0, int(top_k))]

    out = []
    for rank, (val, l, s, all_metrics) in enumerate(items, start=1):
        entry = {
            "rank": rank,
            "layer": l,
            "step": int(s) if s.isdigit() else s,
            f"{metric}": float(val),
        }
        # Include other metrics for context
        for k, v in all_metrics.items():
            if k != metric:
                entry[k] = float(v)
        out.append(entry)
    return out


def save_topk_report(
    topk: List[Dict[str, Union[str, int, float]]],
    out_json: str,
    out_csv: str,
) -> None:
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(topk, f, indent=2)

    if topk:
        fieldnames = list(topk[0].keys())
    else:
        fieldnames = ["rank", "layer", "step", "score"]
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in topk:
            writer.writerow(row)


def _index_per_pair(per_pair: PerPair) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in per_pair:
        pid = str(rec.get("pair_id", ""))
        if pid:
            out[pid] = rec
    return out


def save_per_pair_curves(
    per_pair: PerPair,
    pair_ids: Sequence[str],
    layers: Sequence[str],
    out_dir: str,
    title_prefix: str = "ΔDLIG site scalar per step",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    idx = _index_per_pair(per_pair)

    for pid in pair_ids:
        if pid not in idx:
            raise ValueError(f"pair_id not found in per_pair: {pid}")

        rec = idx[pid]
        delta_dlig = rec.get("delta_dlig", {})
        if not isinstance(delta_dlig, dict):
            raise ValueError(f"per_pair record {pid} has invalid delta_dlig")

        for layer in layers:
            lkey = str(layer)
            if lkey not in delta_dlig:
                raise ValueError(f"pair_id {pid} missing layer {lkey} in delta_dlig")

            smap = delta_dlig[lkey]
            if not isinstance(smap, dict):
                raise ValueError(f"pair_id {pid} layer {lkey} delta_dlig must be dict step->scalar")

            steps_sorted = sorted(list(smap.keys()), key=lambda x: int(x) if str(x).isdigit() else str(x))
            xs = list(range(len(steps_sorted)))
            
            # Handle both old format (step -> float) and new format (step -> {delta:, harm_abs:})
            sample_val = smap[steps_sorted[0]]
            if isinstance(sample_val, dict):
                # New format with multiple metrics
                fig, ax = plt.subplots(figsize=(10, 5))
                
                if "delta" in sample_val:
                    ys_delta = [float(smap[s]["delta"]) for s in steps_sorted]
                    ax.plot(xs, ys_delta, label="ΔDLIG", color="red", marker='o', markersize=3)
                
                if "harm_abs" in sample_val:
                    ys_harm = [float(smap[s]["harm_abs"]) for s in steps_sorted]
                    ax.plot(xs, ys_harm, label="||DLIG_harm||", color="orange", marker='s', markersize=3)
                
                ax.legend(loc="best")
            else:
                # Legacy format
                fig, ax = plt.subplots(figsize=(10, 4))
                ys = [float(smap[s]) for s in steps_sorted]
                ax.plot(xs, ys)

            ax.set_title(f"{title_prefix} | pair={pid} layer={lkey}")
            ax.set_xlabel("diffusion step t")
            ax.set_ylabel("Score")
            ax.set_xticks(xs)
            ax.set_xticklabels([str(s) for s in steps_sorted], rotation=90)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            out_png = os.path.join(out_dir, f"per_pair_{pid}_layer_{lkey}.png")
            out_pdf = os.path.join(out_dir, f"per_pair_{pid}_layer_{lkey}.pdf")
            fig.savefig(out_png, dpi=200)
            fig.savefig(out_pdf)
            plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize contrastive ΔDLIG outputs")

    p.add_argument("--input_dir", type=str, required=True, help="Directory containing runner outputs.")
    p.add_argument("--out_dir", type=str, default="", help="Output directory for figures/reports. Defaults to <input_dir>/viz")

    p.add_argument("--site_scores", type=str, default="", help="Path to site_scores_S_l_t.(json|pt).")
    p.add_argument("--per_pair", type=str, default="", help="Path to per_pair_delta_dlig.(json|pt).")

    p.add_argument("--layers", type=str, nargs="*", default=None, help="Optional subset of layers to plot.")
    p.add_argument("--steps", type=str, nargs="*", default=None, help="Optional subset of steps for top-K.")

    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--bottom_k", type=int, default=0)
    p.add_argument("--topk_metric", type=str, default="delta_mean", help="Metric to use for top-K ranking.")

    p.add_argument("--pair_ids", type=str, nargs="*", default=None, help="Optional pair_ids for per-pair curve plots.")
    
    p.add_argument("--skip_std_bands", action="store_true", help="Skip std band plots.")
    
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    input_dir = str(args.input_dir)
    out_dir = str(args.out_dir) if args.out_dir else os.path.join(input_dir, "viz")
    os.makedirs(out_dir, exist_ok=True)

    site_scores_path = str(args.site_scores) if args.site_scores else os.path.join(input_dir, "site_scores_S_l_t.json")
    if not os.path.exists(site_scores_path):
        alt = os.path.join(input_dir, "site_scores_S_l_t.pt")
        if os.path.exists(alt):
            site_scores_path = alt
        else:
            raise FileNotFoundError(f"Missing site_scores file: {site_scores_path}")

    per_pair_path = str(args.per_pair) if args.per_pair else os.path.join(input_dir, "per_pair_delta_dlig.json")
    if not os.path.exists(per_pair_path):
        alt = os.path.join(input_dir, "per_pair_delta_dlig.pt")
        if os.path.exists(alt):
            per_pair_path = alt
        else:
            per_pair_path = ""

    site_scores, available_metrics = load_site_scores(site_scores_path)
    print(f"[INFO] Available metrics: {available_metrics}")
    
    layers_sorted = _sorted_layers(site_scores, args.layers)
    steps_sorted = _sorted_steps(site_scores, layers_sorted)
    
    print(f"[INFO] Layers: {layers_sorted}")
    print(f"[INFO] Steps: {steps_sorted}")

    # Generate heatmaps for all available metrics
    save_comparison_heatmaps(
        site_scores=site_scores,
        layers_sorted=layers_sorted,
        steps_sorted=steps_sorted,
        metrics=available_metrics,
        out_dir=out_dir,
    )

    # Generate per-layer curves for all metrics
    save_multi_metric_layer_curves(
        site_scores=site_scores,
        layers_sorted=layers_sorted,
        steps_sorted=steps_sorted,
        metrics=available_metrics,
        out_dir=out_dir,
    )
    
    # Std bands plots
    if not args.skip_std_bands:
        save_std_bands_plot(site_scores, layers_sorted, steps_sorted, out_dir, "delta")
        save_std_bands_plot(site_scores, layers_sorted, steps_sorted, out_dir, "harm")

    # Top-K / Bottom-K
    topk_metric = args.topk_metric if args.topk_metric in available_metrics else available_metrics[0]
    print(f"[INFO] Using metric '{topk_metric}' for top-K ranking")
    
    topk = topk_sites(site_scores, top_k=int(args.top_k), metric=topk_metric, layers=layers_sorted, steps=args.steps, largest=True)
    save_topk_report(
        topk,
        out_json=os.path.join(out_dir, f"topk_sites_{topk_metric}.json"),
        out_csv=os.path.join(out_dir, f"topk_sites_{topk_metric}.csv"),
    )

    if int(args.bottom_k) > 0:
        bottomk = topk_sites(site_scores, top_k=int(args.bottom_k), metric=topk_metric, layers=layers_sorted, steps=args.steps, largest=False)
        save_topk_report(
            bottomk,
            out_json=os.path.join(out_dir, f"bottomk_sites_{topk_metric}.json"),
            out_csv=os.path.join(out_dir, f"bottomk_sites_{topk_metric}.csv"),
        )

    # Optional per-pair plots
    if per_pair_path and args.pair_ids is not None and len(args.pair_ids) > 0:
        per_pair = load_per_pair(per_pair_path)
        save_per_pair_curves(
            per_pair=per_pair,
            pair_ids=[str(x) for x in args.pair_ids],
            layers=layers_sorted,
            out_dir=os.path.join(out_dir, "per_pair_curves"),
        )

    print(f"[OK] Wrote visualizations to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()