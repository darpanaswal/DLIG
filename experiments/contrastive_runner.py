"""
Contrastive ΔDLIG experiment runner (Distributional Contrast).

Optimizations over original:
1. Multi-layer activation caching: All layer activations captured in 2 forward
   passes per timestep (real + baseline) instead of 2 per (layer × timestep).
   For 5 layers × 8 steps, this reduces activation-capture forwards from 80 to 16.

2. torchrun / DDP parallelization: Each GPU loads its own model replica and
   processes a disjoint shard of prompts. Accumulators are reduced across ranks
   at the end. Launch with:
       torchrun --nproc_per_node=4 contrastive_runner.py --layers 0 7 14 21 26 ...

   Single-GPU mode still works:
       python contrastive_runner.py --layers 0 7 14 21 26 ...

Notes on integration_batch_size with DDP:
- Each GPU holds a full model replica (~14GB for Dream 7B in bf16).
- On 40GB A100s, ~26GB remains for activations/gradients.
- integration_batch_size=5 is fine in bf16. Lower to 2-3 if using fp32.
"""

import gc
import os
import json
import time
import torch
import torch.distributed as dist
import argparse
import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from attribution.hook_manager import MultiLayerHookManager
from attribution.dlig_attribution import DLIGAttribution
from utils.config import MODEL_PATH, OUTPUT_DIR, CONTRAST_DATASET
from models.model_manager import ModelManager, GradientEnabledModel
from utils.contrastive_utils import (
    compute_absolute_dlig_site_value,
    update_site_score_accumulator,
    finalize_site_scores,
)


# ------------------------------------------------------------------ #
#  DDP helpers
# ------------------------------------------------------------------ #

def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def print_rank0(msg: str, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(msg, **kwargs)


def shard_list(items: list, rank: int, world_size: int) -> list:
    """Return the shard of `items` belonging to `rank`."""
    return items[rank::world_size]


def reduce_site_accumulators(
    local_accum: Dict[Tuple[str, int], Dict[str, Any]],
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    All-reduce site accumulators across ranks.

    Each rank has partial sums. We serialize to a flat dict, gather on rank 0,
    and merge by summing sums/counts and taking max of maxes.
    """
    if not is_dist_initialized() or get_world_size() == 1:
        return local_accum

    serialized = {}
    for (layer, step), v in local_accum.items():
        key = f"{layer}||{step}"
        serialized[key] = v

    gathered = [None] * get_world_size()
    dist.all_gather_object(gathered, serialized)

    merged: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for rank_data in gathered:
        for composite_key, v in rank_data.items():
            layer, step_str = composite_key.split("||")
            key = (layer, int(step_str))
            if key not in merged:
                merged[key] = {}

            for field, value in v.items():
                if field.endswith("_count") or field == "count":
                    merged[key][field] = merged[key].get(field, 0) + value
                elif field.endswith("_max") or field == "max":
                    merged[key][field] = max(merged[key].get(field, float("-inf")), value)
                else:
                    merged[key][field] = merged[key].get(field, 0.0) + value

    return merged


def reduce_all_records(
    local_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Gather per-prompt records from all ranks to rank 0."""
    if not is_dist_initialized() or get_world_size() == 1:
        return local_records

    gathered = [None] * get_world_size()
    dist.all_gather_object(gathered, local_records)

    if is_main_process():
        merged = []
        for rank_records in gathered:
            merged.extend(rank_records)
        return merged
    return local_records


# ------------------------------------------------------------------ #
#  Progress tracker
# ------------------------------------------------------------------ #

class ProgressTracker:
    """
    Tracks per-prompt wall time and prints global ETA after each prompt.

    Output example:
        [PROGRESS] rank=0 | 3/125 done (2.4%) | last=42.3s avg=45.1s | ETA=1h31m | elapsed=2m15s
    """

    def __init__(self, total: int, rank: int = 0, world_size: int = 1):
        self.total = total            # prompts on THIS rank
        self.rank = rank
        self.world_size = world_size
        self.global_total = total * world_size  # approximate
        self.start_time = time.time()
        self.prompt_times: List[float] = []

    def record(self, prompt_dt: float) -> None:
        self.prompt_times.append(prompt_dt)

    def summary(self) -> str:
        done = len(self.prompt_times)
        remaining = self.total - done
        elapsed = time.time() - self.start_time

        last_dt = self.prompt_times[-1]

        # Exponential moving average (alpha=0.3) for more responsive estimates
        if len(self.prompt_times) == 1:
            ema = last_dt
        else:
            alpha = 0.3
            ema = self.prompt_times[0]
            for t in self.prompt_times[1:]:
                ema = alpha * t + (1 - alpha) * ema

        eta_seconds = ema * remaining
        pct = 100.0 * done / self.total if self.total > 0 else 100.0

        return (
            f"[PROGRESS] rank={self.rank} | "
            f"{done}/{self.total} done ({pct:.1f}%) | "
            f"last={last_dt:.1f}s avg={ema:.1f}s | "
            f"ETA={self._fmt_time(eta_seconds)} | "
            f"elapsed={self._fmt_time(elapsed)}"
        )

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.1f}m"
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h{mins:02d}m"


# ------------------------------------------------------------------ #
#  Data structures & utilities
# ------------------------------------------------------------------ #

@dataclass
class Prompt:
    """Single prompt with metadata."""
    prompt_id: str
    user_message: str
    label: str  # "harmful" or "benign"
    system: str = "You are a helpful assistant."


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Hard determinism controls (per-rank seeding for DDP)."""
    rank = get_rank()
    seed = int(seed) + rank

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    harmful_raw = data.get("harmful", [])
    benign_raw = data.get("benign", [])

    if max_harmful is not None:
        harmful_raw = harmful_raw[:max_harmful]
    if max_benign is not None:
        benign_raw = benign_raw[:max_benign]

    harmful_prompts = [
        Prompt(prompt_id=f"harmful_{i:04d}", user_message=msg, label="harmful")
        for i, msg in enumerate(harmful_raw)
    ]
    benign_prompts = [
        Prompt(prompt_id=f"benign_{i:04d}", user_message=msg, label="benign")
        for i, msg in enumerate(benign_raw)
    ]

    print_rank0(f"[INFO] Loaded {len(harmful_prompts)} harmful prompts")
    print_rank0(f"[INFO] Loaded {len(benign_prompts)} benign prompts")

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
    """Records x_t during generation (stored on CPU)."""

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


# ------------------------------------------------------------------ #
#  Core processing (multi-layer caching)
# ------------------------------------------------------------------ #

def process_single_prompt(
    *,
    prompt: Prompt,
    model,
    tokenizer,
    device: torch.device,
    grad_model: GradientEnabledModel,
    multi_hook: MultiLayerHookManager,
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
) -> Dict[str, Any]:
    """
    Process a single prompt with multi-layer activation caching.

    Forward pass savings:
        Original: 2 × |layers| × |steps| activation-capture forwards
        New:      2 × |steps| activation-capture forwards
        For 5 layers × 8 steps: 80 → 16 forwards saved.
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
            print(f"[HEARTBEAT] rank={get_rank()} {prompt.prompt_id} gen step={int(step)}", flush=True)
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

    prefix = "harm_" if prompt.label == "harmful" else "benign_"

    # Get mask token ID
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = pad_id

    # Create per-layer DLIG instances (reuse across steps)
    dlig_instances: Dict[str, DLIGAttribution] = {}
    for layer in layers:
        layer_view = multi_hook.get_layer_view(layer)
        dlig = DLIGAttribution(
            model,
            tokenizer,
            layer_view,
            integration_steps=integration_steps,
            integration_batch_size=integration_batch_size,
            enable_timing=dlig_timing,
            timing_log_every_chunks=dlig_timing_log_every_chunks,
            disable_kv_cache=disable_kv_cache,
        )
        dlig.set_original_input_length(L_prompt)
        dlig.relevant_token_indices = []
        dlig_instances[layer] = dlig

    # Initialize per-layer records
    layer_recs: Dict[str, Dict[str, Any]] = {layer: {} for layer in layers}

    # ---- Main loop: iterate over timesteps, not layers ----
    for step in recorded_steps:
        step_start = time.time()

        x_t = recorder.x_by_step[step].to(device)
        baseline_inp = dlig_instances[layers[0]].create_baseline_input(x_t, int(mask_token_id), L_prompt)

        # ---- ONE forward pass for real activations at ALL layers ----
        all_real_acts = multi_hook.capture_activations(x_t, disable_kv_cache=disable_kv_cache)

        # ---- ONE forward pass for baseline activations at ALL layers ----
        all_baseline_acts = multi_hook.capture_activations(baseline_inp, disable_kv_cache=disable_kv_cache)

        dt_capture = time.time() - step_start

        # ---- Integration per layer (no extra forward passes for capture) ----
        for layer in layers:
            t_layer = time.time()

            real_act = all_real_acts[layer]
            baseline_act = all_baseline_acts[layer]

            dlig = dlig_instances[layer]
            dlig_result = dlig.compute_dlig_at_timestep_with_activations(
                step=step,
                x_t=x_t,
                real_act=real_act,
                baseline_act=baseline_act,
                original_length=L_prompt,
            )

            full_dlig = dlig_result["full_dlig"].to(torch.float32)
            abs_value = compute_absolute_dlig_site_value(full_dlig, norm_type="l2")
            update_site_score_accumulator(site_accum, str(layer), int(step), abs_value, prefix=prefix)

            layer_recs[layer][str(step)] = {
                "abs_value": float(abs_value.mean().item()),
            }

            dt_layer = time.time() - t_layer
            print(
                f"[STEP-TIME] rank={get_rank()} {prompt.prompt_id} layer={layer} step={step} "
                f"capture={dt_capture:.2f}s integration={dt_layer:.2f}s "
                f"|dlig|={layer_recs[layer][str(step)]['abs_value']:.4e}",
                flush=True,
            )

            del real_act, baseline_act, dlig_result, full_dlig

        del all_real_acts, all_baseline_acts, x_t

        if clear_cuda_cache_each_step and torch.cuda.is_available():
            torch.cuda.empty_cache()

    for layer in layers:
        prompt_record["dlig_scores"][str(layer)] = layer_recs[layer]

    del recorder
    gc.collect()

    return prompt_record


# ------------------------------------------------------------------ #
#  Experiment runner (with DDP + progress tracking)
# ------------------------------------------------------------------ #

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
    log_every_steps: int = 1,
    dlig_timing: bool = False,
    dlig_timing_log_every_chunks: int = 1,
    disable_kv_cache: bool = True,
    clear_cuda_cache_each_step: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Run distributional contrastive DLIG experiment with optional DDP.
    """
    set_global_seed(seed, deterministic=True)

    rank = get_rank()
    world_size = get_world_size()

    site_accum: Dict[Tuple[str, int], Dict[str, Any]] = {}
    all_records: List[Dict[str, Any]] = []

    grad_model = GradientEnabledModel(model)

    # ---- Create multi-layer hook manager (registered once) ----
    multi_hook = MultiLayerHookManager(model, layer_specs=layers)

    t0 = time.time()

    all_prompts = harmful_prompts + benign_prompts
    total_prompts = len(all_prompts)

    # Shard prompts across ranks
    my_prompts = shard_list(all_prompts, rank, world_size)

    print_rank0(f"[INFO] Starting distributional contrastive experiment")
    print_rank0(f"[INFO] Total prompts: {total_prompts} ({len(harmful_prompts)} harmful, {len(benign_prompts)} benign)")
    print_rank0(f"[INFO] Layers: {layers} (multi-layer caching enabled)")
    print_rank0(f"[INFO] Generation steps: {generation_steps}")
    print_rank0(f"[INFO] World size: {world_size}, prompts per rank: ~{len(my_prompts)}")

    tracker = ProgressTracker(total=len(my_prompts), rank=rank, world_size=world_size)

    try:
        for i, prompt in enumerate(my_prompts):
            prompt_t0 = time.time()

            print(
                f"\n[INFO] rank={rank} Processing prompt {i+1}/{len(my_prompts)}: "
                f"{prompt.prompt_id} ({prompt.label})",
                flush=True,
            )

            record = process_single_prompt(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                device=device,
                grad_model=grad_model,
                multi_hook=multi_hook,
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
            )

            all_records.append(record)

            # Track and print progress
            prompt_dt = time.time() - prompt_t0
            tracker.record(prompt_dt)
            print(tracker.summary(), flush=True)

            # Periodic checkpoint (rank 0 only)
            if is_main_process() and (i + 1) % 50 == 0:
                print(f"[CHECKPOINT] rank=0 processed {i+1}/{len(my_prompts)} prompts", flush=True)
                intermediate_scores = finalize_site_scores(site_accum)
                save_json(
                    os.path.join(output_dir, f"site_scores_checkpoint_rank0_{i+1}.json"),
                    intermediate_scores,
                )

        # ---- DDP: synchronize and merge accumulators ----
        if is_dist_initialized():
            print(f"[SYNC] rank={rank} waiting at barrier before reduce...", flush=True)
            dist.barrier()

        site_accum = reduce_site_accumulators(site_accum)
        all_records = reduce_all_records(all_records)

        # Finalize (all ranks compute, but only rank 0 saves)
        site_scores = finalize_site_scores(site_accum)

        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)
            save_json(os.path.join(output_dir, "per_prompt_dlig.json"), all_records)
            save_json(os.path.join(output_dir, "site_scores_S_l_t.json"), site_scores)
            torch.save(all_records, os.path.join(output_dir, "per_prompt_dlig.pt"))
            torch.save(site_scores, os.path.join(output_dir, "site_scores_S_l_t.pt"))

            print(f"\n[OK] Saved outputs to: {output_dir}", flush=True)
            print(f"[OK] Total wall time: {time.time() - t0:.1f}s", flush=True)

            print("\n" + "=" * 70)
            print("TOP SITES BY |S_diff|")
            print("=" * 70)

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


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Contrastive ΔDLIG experiment runner (multi-layer caching + DDP)"
    )

    # Model arguments
    p.add_argument("--model_path", type=str, default=str(MODEL_PATH))
    p.add_argument("--torch_dtype", type=str, default="bfloat16",
                   help="Model dtype. Use bfloat16 for ~2x speedup on A100s.")

    # Dataset arguments
    p.add_argument("--dataset_path", type=str, default=str(CONTRAST_DATASET))
    p.add_argument("--max_harmful", type=int, default=None)
    p.add_argument("--max_benign", type=int, default=None)

    # DLIG arguments
    p.add_argument("--layers", type=str, nargs="+", required=True,
                   help="Layers to analyze (e.g., 0 7 14 21 26)")
    p.add_argument("--integration_steps", type=int, default=20)
    p.add_argument("--integration_batch_size", type=int, default=5,
                   help="Chunk size for Riemann sum. 5 is fine for bf16 on 40GB A100. "
                        "Lower to 2-3 for fp32.")

    # Generation arguments
    p.add_argument("--generation_steps", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=256)

    # Output arguments
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=os.path.join(str(OUTPUT_DIR), "contrastive"))

    # Logging arguments
    p.add_argument("--log_every_steps", type=int, default=1)
    p.add_argument("--dlig_timing", action="store_true")
    p.add_argument("--dlig_timing_log_every_chunks", type=int, default=1)

    # Performance arguments
    p.add_argument("--disable_kv_cache", action="store_true")
    p.add_argument("--clear_cuda_cache_each_step", action="store_true")

    return p


def init_distributed():
    """Initialize DDP if launched via torchrun. No-op for single-GPU."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        print(f"[DDP] Initialized rank={rank}/{world_size}, local_rank={local_rank}, "
              f"device=cuda:{local_rank}", flush=True)
        return local_rank
    return 0


def main():
    args = build_arg_parser().parse_args()

    local_rank = init_distributed()

    set_global_seed(args.seed, deterministic=True)

    # Each rank loads its own model replica on its own GPU
    if is_dist_initialized():
        device_map = {"": f"cuda:{local_rank}"}
    else:
        device_map = "auto"

    print_rank0(f"[INFO] Loading model from {args.model_path} (dtype={args.torch_dtype})")
    mm = ModelManager(
        model_path=args.model_path,
        device_map=device_map,
        torch_dtype=args.torch_dtype,
    )
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()
    print(f"[INFO] rank={get_rank()} model loaded on device: {device}", flush=True)

    # Load dataset (all ranks load the same dataset, then shard)
    print_rank0(f"[INFO] Loading dataset from {args.dataset_path}")
    harmful_prompts, benign_prompts = load_contrast_dataset(
        args.dataset_path,
        max_harmful=args.max_harmful,
        max_benign=args.max_benign,
    )

    ibs = int(args.integration_batch_size)
    ibs = None if ibs <= 0 else ibs

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
        log_every_steps=int(args.log_every_steps),
        dlig_timing=bool(args.dlig_timing),
        dlig_timing_log_every_chunks=int(args.dlig_timing_log_every_chunks),
        disable_kv_cache=bool(args.disable_kv_cache),
        clear_cuda_cache_each_step=bool(args.clear_cuda_cache_each_step),
    )

    if is_dist_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()