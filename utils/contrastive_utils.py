# utils/contrastive_utils.py
"""
Utilities for contrastive ΔDLIG computation and site-score aggregation.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

@dataclass(frozen=True)
class SiteKey:
    layer: str
    step: int


def compute_delta_dlig_site_value(
    dlig_harm_full: torch.Tensor,
    dlig_benign_full: torch.Tensor,
    norm_type: str = "l2",
) -> torch.Tensor:
    """
    Compute a scalar per-example site value from ΔDLIG tensors.

    Inputs:
      dlig_*_full: [B, L_prompt, H]

    Output:
      site_value: [B] (norm over prompt tokens and hidden dim)

    Norm definition for site scoring:
      ||ΔDLIG|| := L2 over (prompt_tokens, hidden_dim) by default.
    """
    if dlig_harm_full.shape != dlig_benign_full.shape:
        raise ValueError(f"DLIG shape mismatch: {dlig_harm_full.shape} vs {dlig_benign_full.shape}")

    delta = dlig_harm_full - dlig_benign_full

    if norm_type == "l2":
        return delta.reshape(delta.shape[0], -1).norm(p=2, dim=-1)
    if norm_type == "l1":
        return delta.reshape(delta.shape[0], -1).norm(p=1, dim=-1)
    raise ValueError(f"Unsupported norm_type: {norm_type}")

def compute_absolute_dlig_site_value(
    dlig_full: torch.Tensor,
    norm_type: str = "l2",
) -> torch.Tensor:
    """
    Compute a scalar per-example site value from absolute DLIG tensor.

    Inputs:
      dlig_full: [B, L_prompt, H]

    Output:
      site_value: [B] (norm over prompt tokens and hidden dim)
    """
    if norm_type == "l2":
        return dlig_full.reshape(dlig_full.shape[0], -1).norm(p=2, dim=-1)
    if norm_type == "l1":
        return dlig_full.reshape(dlig_full.shape[0], -1).norm(p=1, dim=-1)
    raise ValueError(f"Unsupported norm_type: {norm_type}")

def update_site_score_accumulator(
    accum: Dict[Tuple[str, int], Dict[str, Any]],
    layer: str,
    step: int,
    site_values: torch.Tensor,
    prefix: str = "",
) -> None:
    """
    Online accumulator for site statistics.

    Tracks: sum, sum_sq, count, max for computing mean, std, max.
    
    Args:
        accum: The accumulator dictionary
        layer: Layer identifier
        step: Timestep
        site_values: Tensor of values to accumulate
        prefix: Optional prefix for keys (e.g., "delta_" or "harm_")
    """
    key = (layer, step)
    if key not in accum:
        accum[key] = {}
    
    # Initialize fields with prefix if not present
    sum_key = f"{prefix}sum"
    sum_sq_key = f"{prefix}sum_sq"
    count_key = f"{prefix}count"
    max_key = f"{prefix}max"
    
    if sum_key not in accum[key]:
        accum[key][sum_key] = 0.0
        accum[key][sum_sq_key] = 0.0
        accum[key][count_key] = 0
        accum[key][max_key] = float("-inf")
    
    accum[key][sum_key] += float(site_values.sum().item())
    accum[key][sum_sq_key] += float((site_values ** 2).sum().item())
    accum[key][count_key] += int(site_values.numel())
    accum[key][max_key] = max(accum[key][max_key], float(site_values.max().item()))

def finalize_site_scores(
    accum: Dict[Tuple[str, int], Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Returns nested dict: scores[layer][step_str] = {
        "delta_mean": ..., "delta_std": ..., "delta_max": ...,
        "harm_mean": ..., "harm_std": ..., "harm_max": ...,
    }
    
    Handles both prefixed (delta_, harm_) and legacy unprefixed accumulators.
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for (layer, step), v in accum.items():
        out.setdefault(str(layer), {})
        step_data: Dict[str, Any] = {}
        
        # Process each prefix that might be present
        prefixes_found = set()
        for key in v.keys():
            if key.endswith("_sum"):
                prefix = key[:-4]  # Remove "_sum"
                if prefix:
                    prefixes_found.add(prefix)
                else:
                    prefixes_found.add("")  # Legacy unprefixed
        
        # If no prefixes found, check for legacy format
        if not prefixes_found and "sum" in v:
            prefixes_found.add("")
        
        for prefix in prefixes_found:
            sum_key = f"{prefix}sum" if prefix else "sum"
            sum_sq_key = f"{prefix}sum_sq" if prefix else "sum_sq"
            count_key = f"{prefix}count" if prefix else "count"
            max_key = f"{prefix}max" if prefix else "max"
            
            if sum_key not in v:
                continue
                
            count = max(v.get(count_key, 1), 1)
            total = v.get(sum_key, 0.0)
            total_sq = v.get(sum_sq_key, 0.0)
            max_val = v.get(max_key, 0.0)
            
            mean = total / count
            variance = (total_sq / count) - (mean ** 2)
            std = (variance ** 0.5) if variance > 0 else 0.0
            
            # Output key prefix (remove trailing underscore for cleaner output)
            out_prefix = prefix.rstrip("_") + "_" if prefix else ""
            
            step_data[f"{out_prefix}mean"] = float(mean)
            step_data[f"{out_prefix}std"] = float(std)
            step_data[f"{out_prefix}max"] = float(max_val)
        
        out[str(layer)][str(step)] = step_data
    
    return out