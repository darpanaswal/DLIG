"""
Contrastive ΔDLIG experiment runner (Distributional Contrast).

Includes:
- Determinism controls in set_global_seed() to support self-check D.
- Removed Attack Success Scoring.
- Loads harmful/benign prompts from JSON dataset file.
"""

import gc
import os
import json
import time
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from attribution.hook_manager import HookManager
from typing import Any, Dict, List, Tuple, Optional
from attribution.dlig_attribution import DLIGAttribution
from utils.config import MODEL_PATH, OUTPUT_DIR, CONTRAST_DATASET
from models.model_manager import ModelManager, GradientEnabledModel
from utils.contrastive_utils import (
    compute_absolute_dlig_site_value,
    update_site_score_accumulator,
    finalize_site_scores
)


@dataclass
class Prompt:
    """Single prompt with metadata."""
    prompt_id: str
    user_message: str
    label: str  # "harmful" or "benign"
    system: str = "You are a helpful assistant."


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Hard determinism controls.

    Notes:
    - Some custom model sampling paths may still be nondeterministic despite this,
      but this is the strongest practical setting for CUDA determinism.
    """
    seed = int(seed)

    # Python / NumPy
    random.seed(seed)
    np.random.seed(seed)

    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CUDA matmul determinism
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # Ensure deterministic algorithms (will error if an op can't be deterministic)
        torch.use_deterministic_algorithms(True, warn_only=True)

        # Reduce nondeterminism from TF32 (A100)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # cuDNN deterministic behavior (mostly relevant for convs, but safe)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Try to force deterministic SDPA behavior if available (PyTorch 2.x)
        try:
            torch.backends.cuda.enable_flash_sdp(False)
        except Exception:
            pass
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        except Exception:
            pass
        try:
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass


def load_contrast_dataset(
    dataset_path: str,
    max_harmful: Optional[int] = None,
    max_benign: Optional[int] = None,
) -> Tuple[List[Prompt], List[Prompt]]:
    """
    Load harmful and benign prompts from JSON file.
    
    Expected JSON format:
    {
        "harmful": ["prompt1", "prompt2", ...],
        "benign": ["prompt1", "prompt2", ...]
    }
    
    Args:
        dataset_path: Path to JSON file
        max_harmful: Maximum number of harmful prompts to load (None = all)
        max_benign: Maximum number of benign prompts to load (None = all)
    
    Returns:
        Tuple of (harmful_prompts, benign_prompts) as lists of Prompt objects
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    harmful_raw = data.get("harmful", [])
    benign_raw = data.get("benign", [])
    
    if max_harmful is not None:
        harmful_raw = harmful_raw[:max_harmful]
    if max_benign is not None:
        benign_raw = benign_raw[:max_benign]
    
    harmful_prompts = [
        Prompt(
            prompt_id=f"harmful_{i:04d}",
            user_message=msg,
            label="harmful",
        )
        for i, msg in enumerate(harmful_raw)
    ]
    
    benign_prompts = [
        Prompt(
            prompt_id=f"benign_{i:04d}",
            user_message=msg,
            label="benign",
        )
        for i, msg in enumerate(benign_raw)
    ]
    
    print(f"[INFO] Loaded {len(harmful_prompts)} harmful prompts")
    print(f"[INFO] Loaded {len(benign_prompts)} benign prompts")
    
    return harmful_prompts, benign_prompts


def resolve_layer_module(model, layer: str):
    if layer.isdigit():
        return model.model.layers[int(layer)]
    if layer == "embed_tokens":
        return model.model.embed_tokens
    raise ValueError(f"Unsupported layer spec: {layer}")


def make_messages(system_text: str, user_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


class TrajectoryRecorder:
    """
    Records x_t during generation (stored on CPU).
    """
    def __init__(self):
        self.x_by_step: Dict[int, torch.Tensor] = {}

    def hook(self, step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if step is None:
            return logits
        self.x_by_step[int(step)] = x.detach().to("cpu", non_blocking=False).clone()
        return logits

    def steps(self) -> List[int]:
        return sorted(self.x_by_step.keys())


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def process_single_prompt(
    *,
    prompt: Prompt,
    model,
    tokenizer,
    device: torch.device,
    grad_model: GradientEnabledModel,
    layers: List[str],
    generation_steps: int,
    max_new_tokens: int,
    integration_steps: int,
    integration_batch_size: Optional[int],
    site_accum: Dict[Tuple[str, int], Dict[str, Any]],
    dlig_timing: bool = False,
    dlig_timing_log_every_chunks: int = 1,
    disable_kv_cache: bool = True,
    clear_cuda_cache_each_step: bool = False,
    log_every_steps: int = 1,
    tqdm_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Process a single prompt: generate trajectory, compute DLIG at all sites, update accumulators.
    
    Returns:
        Dictionary with prompt metadata and per-layer/step DLIG scores
    """
    # Tokenize
    inputs = tokenizer.apply_chat_template(
        make_messages(prompt.system, prompt.user_message),
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device).float()
    L_prompt = input_ids.shape[1]
    
    # Record trajectory
    recorder = TrajectoryRecorder()
    
    def trajectory_hook(step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        out = recorder.hook(step, x, logits)
        if step is not None and log_every_steps > 0 and (int(step) % log_every_steps == 0):
            print(f"[HEARTBEAT] {prompt.prompt_id} gen step={int(step)}", flush=True)
        return out
    
    _ = grad_model.diffusion_generate_with_grad(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        steps=generation_steps,
        generation_logits_hook_func=trajectory_hook,
    )
    
    recorded_steps = recorder.steps()
    
    # Decode final text
    last_step = max(recorded_steps) if recorded_steps else None
    final_x = recorder.x_by_step[last_step] if last_step is not None else input_ids.detach().cpu()
    generated_text = tokenizer.decode(final_x[0, L_prompt:], skip_special_tokens=True)
    
    # Prepare output record
    prompt_record: Dict[str, Any] = {
        "prompt_id": prompt.prompt_id,
        "label": prompt.label,
        "user_message": prompt.user_message,
        "prompt_length": int(L_prompt),
        "generated_text": generated_text,
        "dlig_scores": {},
    }
    
    # Determine prefix for accumulator
    prefix = "harm_" if prompt.label == "harmful" else "benign_"
    
    # Get mask token ID
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = pad_id
    
    # Process each layer
    layer_iter = tqdm(layers, desc=f"Layers({prompt.prompt_id})", leave=False, disable=not tqdm_enabled)
    for layer in layer_iter:
        target_layer_module = resolve_layer_module(model, layer)
        hook_manager = HookManager(model)
        hook_manager.register_hook(target_layer_module)
        
        dlig = DLIGAttribution(
            model,
            tokenizer,
            hook_manager,
            integration_steps=integration_steps,
            integration_batch_size=integration_batch_size,
            enable_timing=dlig_timing,
            timing_log_every_chunks=dlig_timing_log_every_chunks,
            disable_kv_cache=disable_kv_cache,
        )
        dlig.set_original_input_length(L_prompt)
        dlig.relevant_token_indices = []
        
        layer_rec: Dict[str, Any] = {}
        
        step_iter = tqdm(recorded_steps, desc=f"Steps(l={layer})", leave=False, disable=not tqdm_enabled)
        for step in step_iter:
            step_start = time.time()
            
            x_t = recorder.x_by_step[step].to(device)
            
            # Compute DLIG
            dlig_result = dlig.compute_dlig_at_timestep(
                step=step,
                x_t=x_t,
                mask_token_id=int(mask_token_id),
                original_length=L_prompt,
            )
            
            full_dlig = dlig_result["full_dlig"].to(torch.float32)
            abs_value = compute_absolute_dlig_site_value(full_dlig, norm_type="l2")
            
            # Update accumulator with appropriate prefix
            update_site_score_accumulator(site_accum, str(layer), int(step), abs_value, prefix=prefix)
            
            # Store per-step value
            layer_rec[str(step)] = {
                "abs_value": float(abs_value.mean().item()),
            }
            
            dt = time.time() - step_start
            print(
                f"[STEP-TIME] {prompt.prompt_id} layer={layer} step={step} time={dt:.2f}s "
                f"|dlig|={layer_rec[str(step)]['abs_value']:.4e}",
                flush=True,
            )
            
            if clear_cuda_cache_each_step and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            del x_t, dlig_result, full_dlig
        
        prompt_record["dlig_scores"][str(layer)] = layer_rec
        hook_manager.remove_hook()
    
    # Cleanup
    del recorder
    gc.collect()
    
    return prompt_record


def run_contrastive_experiment(
    *,
    model,
    tokenizer,
    device: torch.device,
    harmful_prompts: List[Prompt],
    benign_prompts: List[Prompt],
    layers: List[str],
    generation_steps: int,
    max_new_tokens: int,
    integration_steps: int,
    integration_batch_size: Optional[int],
    seed: int,
    output_dir: str,
    tqdm_enabled: bool = True,
    log_every_steps: int = 1,
    dlig_timing: bool = False,
    dlig_timing_log_every_chunks: int = 1,
    disable_kv_cache: bool = True,
    clear_cuda_cache_each_step: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Run distributional contrastive DLIG experiment.
    
    Processes all harmful and benign prompts independently, accumulates statistics,
    and computes S_diff = S_harm - S_benign for each (layer, timestep) site.
    """
    set_global_seed(seed, deterministic=True)
    
    site_accum: Dict[Tuple[str, int], Dict[str, Any]] = {}
    all_records: List[Dict[str, Any]] = []
    
    grad_model = GradientEnabledModel(model)
    
    t0 = time.time()
    
    # Combine all prompts for processing
    all_prompts = harmful_prompts + benign_prompts
    total_prompts = len(all_prompts)
    
    print(f"[INFO] Starting distributional contrastive experiment")
    print(f"[INFO] Total prompts: {total_prompts} ({len(harmful_prompts)} harmful, {len(benign_prompts)} benign)")
    print(f"[INFO] Layers: {layers}")
    print(f"[INFO] Generation steps: {generation_steps}")
    
    try:
        prompt_iter = tqdm(all_prompts, desc="Prompts", disable=not tqdm_enabled)
        
        for i, prompt in enumerate(prompt_iter):
            print(f"\n[INFO] Processing prompt {i+1}/{total_prompts}: {prompt.prompt_id} ({prompt.label})", flush=True)
            
            record = process_single_prompt(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                device=device,
                grad_model=grad_model,
                layers=layers,
                generation_steps=generation_steps,
                max_new_tokens=max_new_tokens,
                integration_steps=integration_steps,
                integration_batch_size=integration_batch_size,
                site_accum=site_accum,
                dlig_timing=dlig_timing,
                dlig_timing_log_every_chunks=dlig_timing_log_every_chunks,
                disable_kv_cache=disable_kv_cache,
                clear_cuda_cache_each_step=clear_cuda_cache_each_step,
                log_every_steps=log_every_steps,
                tqdm_enabled=tqdm_enabled,
            )
            
            all_records.append(record)
            
            # Periodic checkpoint
            if (i + 1) % 50 == 0:
                print(f"[CHECKPOINT] Processed {i+1}/{total_prompts} prompts", flush=True)
                # Save intermediate results
                intermediate_scores = finalize_site_scores(site_accum)
                save_json(
                    os.path.join(output_dir, f"site_scores_checkpoint_{i+1}.json"),
                    intermediate_scores
                )
        
        # Finalize site scores
        site_scores = finalize_site_scores(site_accum)
        
        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        save_json(os.path.join(output_dir, "per_prompt_dlig.json"), all_records)
        save_json(os.path.join(output_dir, "site_scores_S_l_t.json"), site_scores)
        torch.save(all_records, os.path.join(output_dir, "per_prompt_dlig.pt"))
        torch.save(site_scores, os.path.join(output_dir, "site_scores_S_l_t.pt"))
        
        # Print summary
        print(f"\n[OK] Saved outputs to: {output_dir}", flush=True)
        print(f"[OK] Total wall time: {time.time() - t0:.1f}s", flush=True)
        
        # Print top sites by |S_diff|
        print("\n" + "="*70)
        print("TOP SITES BY |S_diff|")
        print("="*70)
        
        site_diffs = []
        for layer, steps in site_scores.items():
            for step, scores in steps.items():
                if "diff_mean" in scores:
                    site_diffs.append({
                        "layer": layer,
                        "step": step,
                        "diff_mean": scores["diff_mean"],
                        "diff_abs_mean": scores["diff_abs_mean"],
                        "harm_mean": scores.get("harm_mean", 0),
                        "benign_mean": scores.get("benign_mean", 0),
                    })
        
        site_diffs.sort(key=lambda x: x["diff_abs_mean"], reverse=True)
        
        for i, site in enumerate(site_diffs[:10]):
            print(
                f"  {i+1}. Layer {site['layer']}, Step {site['step']}: "
                f"S_diff={site['diff_mean']:.4e} "
                f"(harm={site['harm_mean']:.4e}, benign={site['benign_mean']:.4e})"
            )
        
        return all_records, site_scores
    
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Contrastive ΔDLIG experiment runner (distributional contrast)")
    
    # Model arguments
    p.add_argument("--model_path", type=str, default=str(MODEL_PATH))
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--torch_dtype", type=str, default="float32")
    
    # Dataset arguments
    p.add_argument("--dataset_path", type=str, default=str(CONTRAST_DATASET),
                   help="Path to JSON file with harmful/benign prompts")
    p.add_argument("--max_harmful", type=int, default=None,
                   help="Maximum number of harmful prompts to use (default: all)")
    p.add_argument("--max_benign", type=int, default=None,
                   help="Maximum number of benign prompts to use (default: all)")
    
    # DLIG arguments
    p.add_argument("--layers", type=str, nargs="+", required=True,
                   help="Layers to analyze (e.g., 0 7 14 21 26)")
    p.add_argument("--integration_steps", type=int, default=20)
    p.add_argument("--integration_batch_size", type=int, default=5,
                   help="0 => full batch (m). Use small values to avoid OOM.")
    
    # Generation arguments
    p.add_argument("--generation_steps", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=256)
    
    # Output arguments
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=os.path.join(str(OUTPUT_DIR), "contrastive"))
    
    # Logging arguments
    p.add_argument("--tqdm", action="store_true")
    p.add_argument("--log_every_steps", type=int, default=1)
    p.add_argument("--dlig_timing", action="store_true")
    p.add_argument("--dlig_timing_log_every_chunks", type=int, default=1)
    
    # Performance arguments
    p.add_argument("--disable_kv_cache", action="store_true")
    p.add_argument("--clear_cuda_cache_each_step", action="store_true")
    
    return p


def main():
    args = build_arg_parser().parse_args()
    
    set_global_seed(args.seed, deterministic=True)
    
    # Load model
    print(f"[INFO] Loading model from {args.model_path}")
    mm = ModelManager(model_path=args.model_path, device_map=args.device_map, torch_dtype=args.torch_dtype)
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()
    print(f"[INFO] Model loaded on device: {device}")
    
    # Load dataset
    print(f"[INFO] Loading dataset from {args.dataset_path}")
    harmful_prompts, benign_prompts = load_contrast_dataset(
        args.dataset_path,
        max_harmful=args.max_harmful,
        max_benign=args.max_benign,
    )
    
    # Parse integration batch size
    ibs = int(args.integration_batch_size)
    ibs = None if ibs <= 0 else ibs
    
    # Run experiment
    run_contrastive_experiment(
        model=model,
        tokenizer=tokenizer,
        device=device,
        harmful_prompts=harmful_prompts,
        benign_prompts=benign_prompts,
        layers=list(args.layers),
        generation_steps=int(args.generation_steps),
        max_new_tokens=int(args.max_new_tokens),
        integration_steps=int(args.integration_steps),
        integration_batch_size=ibs,
        seed=int(args.seed),
        output_dir=str(args.output_dir),
        tqdm_enabled=bool(args.tqdm),
        log_every_steps=int(args.log_every_steps),
        dlig_timing=bool(args.dlig_timing),
        dlig_timing_log_every_chunks=int(args.dlig_timing_log_every_chunks),
        disable_kv_cache=bool(args.disable_kv_cache),
        clear_cuda_cache_each_step=bool(args.clear_cuda_cache_each_step),
    )


if __name__ == "__main__":
    main()