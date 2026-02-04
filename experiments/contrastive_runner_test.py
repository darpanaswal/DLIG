# experiments/contrastive_runner_test.py
"""
Self-check harness (A–E) for the contrastive ΔDLIG pipeline.

Hard-fail checks:
B) Identity-pair sanity: harm == benign => ΔDLIG ~ 0
C) Same-x_t suffix invariance (within-run): x_t_harm[:, L_prompt:] == x_t_benign[:, L_prompt:]
D) Run sanity (within-run): recorded step indices exist and are monotonic
E) DLIG semantics:
   - baseline masks only [:L_prompt]
   - F_t equals manual recomputation over logits[:, L_prompt:]

NOTE on cross-run nondeterminism:
- Some diffusion samplers / GPU attention paths are inherently nondeterministic across runs.
- Therefore we do NOT hard-fail on cross-run equality of x_t or ΔDLIG.
- We optionally PRINT a variability report if --report_variability is set.
"""

import os
import torch
import argparse
from typing import List, Dict
from utils.config import MODEL_PATH, OUTPUT_DIR
from attribution.hook_manager import HookManager
from attribution.dlig_attribution import DLIGAttribution
from models.model_manager import ModelManager, GradientEnabledModel
from experiments.contrastive_runner import (
    PromptPair,
    build_default_pairs,
    make_messages,
    pad_to_same_length,
    resolve_layer_module,
    set_global_seed,
)
from utils.contrastive_utils import compute_delta_dlig_site_value


def assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, msg: str) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{msg} (shape mismatch) {a.shape} vs {b.shape}")
    if not torch.equal(a, b):
        diff = (a != b).to(torch.int32).sum().item()
        raise AssertionError(f"{msg} (value mismatch) differing elements: {diff}")


def pick_check_steps(all_steps: List[int], k: int) -> List[int]:
    if not all_steps or k <= 0:
        return []
    if len(all_steps) <= k:
        return all_steps
    idxs = [0, len(all_steps) // 2, len(all_steps) - 1]
    out = []
    for i in idxs:
        if all_steps[i] not in out:
            out.append(all_steps[i])
    return out[:k]


class TrajectoryRecorderCPU:
    def __init__(self):
        self.x_by_step: Dict[int, torch.Tensor] = {}

    def hook(self, step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if step is None:
            return logits
        self.x_by_step[int(step)] = x.detach().to("cpu", non_blocking=False).clone()
        return logits

    def steps(self) -> List[int]:
        return sorted(self.x_by_step.keys())


def _model_forward_no_mask(model, input_ids: torch.Tensor, disable_kv_cache: bool):
    if disable_kv_cache:
        try:
            return model(input_ids=input_ids, attention_mask=None, use_cache=False)
        except TypeError:
            return model(input_ids=input_ids, attention_mask=None)
    return model(input_ids=input_ids, attention_mask=None)


def semantics_spot_check(
    dlig: DLIGAttribution,
    model,
    x_t: torch.Tensor,
    L_prompt: int,
    mask_token_id: int,
    disable_kv_cache: bool,
) -> None:
    baseline = dlig.create_baseline_input(x_t, mask_token_id, L_prompt)
    if not torch.all(baseline[:, :L_prompt] == mask_token_id):
        raise AssertionError("E-check failed: baseline did not mask [:L_prompt].")
    assert_tensor_equal(
        baseline[:, L_prompt:],
        x_t[:, L_prompt:],
        "E-check failed: baseline changed suffix tokens after L_prompt.",
    )

    with torch.no_grad():
        outputs = _model_forward_no_mask(model, x_t, disable_kv_cache=disable_kv_cache)

    score_dlig = dlig._compute_target_score(outputs, x_t, L_prompt)

    logits = outputs.logits
    gen_logits = logits[:, L_prompt:, :]
    log_probs = torch.log_softmax(gen_logits, dim=-1)
    max_log_probs, _ = log_probs.max(dim=-1)
    score_manual = max_log_probs.sum()

    if not torch.allclose(score_dlig, score_manual, rtol=0.0, atol=0.0):
        raise AssertionError("E-check failed: F_t mismatch between DLIG and manual recomputation.")


def _compute_site_scalar(
    *,
    dlig: DLIGAttribution,
    x_t_harm: torch.Tensor,
    benign_ids: torch.Tensor,
    L_prompt: int,
    mask_token_id: int,
    step: int,
) -> float:
    x_t_benign = x_t_harm.clone()
    x_t_benign[:, :L_prompt] = benign_ids[:, :L_prompt]

    dlig_h = dlig.compute_dlig_at_timestep(
        step=step,
        x_t=x_t_harm,
        logits=None,
        mask_token_id=int(mask_token_id),
        original_length=L_prompt,
        attention_mask=None,
    )
    dlig_b = dlig.compute_dlig_at_timestep(
        step=step,
        x_t=x_t_benign,
        logits=None,
        mask_token_id=int(mask_token_id),
        original_length=L_prompt,
        attention_mask=None,
    )

    harm_full = dlig_h["full_dlig"].to(torch.float32)
    benign_full = dlig_b["full_dlig"].to(torch.float32)
    site_values = compute_delta_dlig_site_value(harm_full, benign_full, norm_type="l2")
    return float(site_values.mean().item())


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Contrastive ΔDLIG self-check harness (A–E)")

    p.add_argument("--model_path", type=str, default=str(MODEL_PATH))
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--torch_dtype", type=str, default="float32")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--layers", type=str, nargs="+", default=["10"])
    p.add_argument("--generation_steps", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--integration_steps", type=int, default=2)
    p.add_argument("--integration_batch_size", type=int, default=0)

    p.add_argument("--check_steps", type=int, default=3)
    p.add_argument("--delta_atol", type=float, default=1e-5)
    p.add_argument("--delta_rtol", type=float, default=1e-3)

    p.add_argument("--disable_kv_cache", action="store_true")
    p.add_argument("--clear_cuda_cache_each_step", action="store_true")

    # NEW: report-only cross-run variability (no assertions)
    p.add_argument("--report_variability", action="store_true", help="Print cross-run ΔDLIG site scalar variability.")

    p.add_argument("--output_dir", type=str, default=os.path.join(str(OUTPUT_DIR), "contrastive_selfcheck_tmp"))
    return p


def main():
    args = build_arg_parser().parse_args()
    set_global_seed(int(args.seed), deterministic=True)

    mm = ModelManager(model_path=args.model_path, device_map=args.device_map, torch_dtype=args.torch_dtype)
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()

    grad_model = GradientEnabledModel(model)

    base_pairs = build_default_pairs()
    if not base_pairs:
        raise RuntimeError("No default pairs available for self-check.")

    p0 = base_pairs[0]
    pairs = [
        PromptPair(
            pair_id="selfcheck_identity_000",
            harmful_user=p0.harmful_user,
            benign_user=p0.harmful_user,
            system=p0.system,
        ),
        base_pairs[0],
    ]

    ibs = int(args.integration_batch_size)
    ibs = None if ibs <= 0 else ibs

    for pair in pairs:
        harm_inputs = tokenizer.apply_chat_template(
            make_messages(pair.system, pair.harmful_user),
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        benign_inputs = tokenizer.apply_chat_template(
            make_messages(pair.system, pair.benign_user),
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )

        harm_ids = harm_inputs.input_ids.to(device)
        harm_mask = harm_inputs.attention_mask.to(device).float()
        benign_ids = benign_inputs.input_ids.to(device)
        benign_mask = benign_inputs.attention_mask.to(device).float()

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        harm_ids, harm_mask, benign_ids, benign_mask, L_prompt = pad_to_same_length(
            harm_ids, harm_mask, benign_ids, benign_mask, pad_id
        )

        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            mask_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else pad_id

        # Run 1 trajectory
        recorder = TrajectoryRecorderCPU()
        _ = grad_model.diffusion_generate_with_grad(
            harm_ids,
            attention_mask=harm_mask,
            max_new_tokens=int(args.max_new_tokens),
            steps=int(args.generation_steps),
            generation_logits_hook_func=recorder.hook,
        )
        steps_list = recorder.steps()
        if not steps_list:
            raise AssertionError("D-check failed: no recorded timesteps.")
        if steps_list != sorted(steps_list):
            raise AssertionError("D-check failed: recorded steps are not monotonic.")

        check_steps = pick_check_steps(steps_list, int(args.check_steps))

        # Optional: Run 2 trajectory for reporting only
        recorder2 = None
        if bool(args.report_variability):
            set_global_seed(int(args.seed), deterministic=True)
            recorder2 = TrajectoryRecorderCPU()
            _ = grad_model.diffusion_generate_with_grad(
                harm_ids,
                attention_mask=harm_mask,
                max_new_tokens=int(args.max_new_tokens),
                steps=int(args.generation_steps),
                generation_logits_hook_func=recorder2.hook,
            )

        for layer in list(args.layers):
            target_layer_module = resolve_layer_module(model, layer)
            hook_manager = HookManager(model)
            hook_manager.register_hook(target_layer_module)

            dlig = DLIGAttribution(
                model,
                tokenizer,
                hook_manager,
                integration_steps=int(args.integration_steps),
                integration_batch_size=ibs,
                enable_timing=False,
                disable_kv_cache=bool(args.disable_kv_cache),
            )
            dlig.set_original_input_length(L_prompt)
            dlig.relevant_token_indices = []

            # E-check on first checked step
            x0 = recorder.x_by_step[check_steps[0]].to(device)
            semantics_spot_check(
                dlig=dlig,
                model=model,
                x_t=x0,
                L_prompt=L_prompt,
                mask_token_id=int(mask_token_id),
                disable_kv_cache=bool(args.disable_kv_cache),
            )
            del x0

            for step in check_steps:
                x_t_harm = recorder.x_by_step[step].to(device)
                x_t_benign = x_t_harm.clone()
                x_t_benign[:, :L_prompt] = benign_ids[:, :L_prompt]

                # C-check (critical)
                assert_tensor_equal(
                    x_t_harm[:, L_prompt:].detach().cpu(),
                    x_t_benign[:, L_prompt:].detach().cpu(),
                    f"C-check failed: suffix mismatch at step {step}.",
                )

                # B-check via full ΔDLIG tensor
                dlig_h = dlig.compute_dlig_at_timestep(
                    step=step,
                    x_t=x_t_harm,
                    logits=None,
                    mask_token_id=int(mask_token_id),
                    original_length=L_prompt,
                    attention_mask=None,
                )
                dlig_b = dlig.compute_dlig_at_timestep(
                    step=step,
                    x_t=x_t_benign,
                    logits=None,
                    mask_token_id=int(mask_token_id),
                    original_length=L_prompt,
                    attention_mask=None,
                )
                harm_full = dlig_h["full_dlig"].to(torch.float32)
                benign_full = dlig_b["full_dlig"].to(torch.float32)
                delta = harm_full - benign_full

                if pair.pair_id.startswith("selfcheck_identity_"):
                    delta_norm = float(delta.reshape(delta.shape[0], -1).norm(p=2, dim=-1).mean().item())
                    harm_norm = float(harm_full.reshape(harm_full.shape[0], -1).norm(p=2, dim=-1).mean().item())
                    thresh = float(args.delta_atol) + float(args.delta_rtol) * max(harm_norm, 1e-12)
                    if delta_norm > thresh:
                        raise AssertionError(
                            f"B-check failed: identity-pair ΔDLIG not ~0 "
                            f"(layer={layer}, step={step}) delta_norm={delta_norm:.6e} "
                            f"harm_norm={harm_norm:.6e} thresh={thresh:.6e}"
                        )

                if bool(args.report_variability) and (recorder2 is not None) and (step in recorder2.x_by_step):
                    s1 = _compute_site_scalar(
                        dlig=dlig,
                        x_t_harm=x_t_harm,
                        benign_ids=benign_ids,
                        L_prompt=L_prompt,
                        mask_token_id=int(mask_token_id),
                        step=int(step),
                    )
                    x2 = recorder2.x_by_step[step].to(device)
                    s2 = _compute_site_scalar(
                        dlig=dlig,
                        x_t_harm=x2,
                        benign_ids=benign_ids,
                        L_prompt=L_prompt,
                        mask_token_id=int(mask_token_id),
                        step=int(step),
                    )
                    ratio = (s1 / s2) if abs(s2) > 1e-12 else float("inf")
                    print(f"[VAR] pair={pair.pair_id} layer={layer} step={step} site1={s1:.4e} site2={s2:.4e} ratio={ratio:.3f}", flush=True)
                    del x2

                if bool(args.clear_cuda_cache_each_step) and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                del x_t_harm, x_t_benign

            hook_manager.remove_hook()

        print(f"[SELF-CHECK OK] pair={pair.pair_id} steps_checked={check_steps} layers={list(args.layers)}", flush=True)

    print("[SELF-CHECK OK] All checks passed.", flush=True)


if __name__ == "__main__":
    main()