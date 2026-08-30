# experiments/contrastive_attribution.py
"""
contrastive_attribution.py — dataset-scale DLIG attribution over (layer, timestep).

Generates one diffusion trajectory per prompt, then computes DLIG per (layer, step)
using MultiLayerHookManager (all layers captured in 2 forward passes per step).

Scoring is always logprob. Target is toggled:
  - --target "<string>" : fixed/contrastive target (e.g. a refusal string).
  - --target omitted    : self-generated target (model's own trajectory tokens).

Output is resumable JSONL, one record per prompt.
"""

import os
import gc
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from models.backends import build_backend
from utils.config import OUTPUT_DIR, CONTRAST_DATASET
from attribution.dlig_attribution import DLIGAttribution
from attribution.hook_manager import MultiLayerHookManager
from models.model_manager import ModelManager, GradientEnabledModel
from experiments.theorems.verify_completeness import TrajRecorder, set_seed, build_prompt_inputs


def clean_token(tok: str) -> str:
    return tok.replace("\u0120", "_").replace("\u2581", "_").replace("\n", "\\n").replace("\t", "\\t")


def input_token_indices(ids, tokenizer, user_prompt=None):
    """Isolates user prompt tokens by sequence-matching, ignoring chat templates."""
    ids = [int(t) for t in ids]
    special_ids = set(tokenizer.all_special_ids)
    if user_prompt:
        prompt_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
        if prompt_ids:
            target_seq = prompt_ids[1:] if len(prompt_ids) > 1 else prompt_ids
            seq_len = len(target_seq)
            for i in range(len(ids) - seq_len, -1, -1):
                if ids[i:i + seq_len] == target_seq:
                    start_idx = i - (1 if len(prompt_ids) > 1 else 0)
                    return [idx for idx in range(start_idx, start_idx + len(prompt_ids))
                            if ids[idx] not in special_ids]
    return []


def build_arg_parser():
    p = argparse.ArgumentParser(description="Dataset-scale DLIG attribution over (layer, timestep)")
    p.add_argument("--family", type=str, default="dream",
                   choices=["dream", "diffugpt"],
                   help="Model family / backend.")
    p.add_argument("--dataset", type=str, default=str(CONTRAST_DATASET))
    p.add_argument("--out_file", type=str, default=OUTPUT_DIR / "contrast/dataset_attribution_results.jsonl")

    # Sharding
    p.add_argument("--num_shards", type=int, default=1, help="Total number of parallel jobs.")
    p.add_argument("--shard_id", type=int, default=0, help="Index of this specific job (0 to num_shards-1).")

    # Sampling
    p.add_argument("--n_per_class", type=int, default=50,
                   help="Prompts to draw per class (harmful/benign). -1 => all.")
    p.add_argument("--system", type=str, default="You are a helpful assistant.")

    # Target toggle
    p.add_argument("--target", type=str, default=None,
                   help="Fixed target string for contrastive attribution. Omit for self-generated.")

    # DLIG hyperparameters
    p.add_argument("--m", type=int, default=12, help="Integration steps. Lower = faster.")
    p.add_argument("--chunk", type=int, default=12, help="Integration batch size (VRAM control).")
    p.add_argument("--gen_steps", type=int, default=12, help="Diffusion generation steps.")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--target_steps", type=int, nargs="+", default=[5],
                   help="Sparse timesteps to analyze.")
    p.add_argument("--layers", type=str, nargs="+",
                   default=[str(i) for i in range(26)])
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    # Modify output file name if sharding
    if args.num_shards > 1:
        out_path = Path(args.out_file)
        args.out_file = str(out_path.parent / f"{out_path.stem}_shard{args.shard_id}{out_path.suffix}")

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    # Load dataset
    with open(args.dataset, "r") as f:
        data = json.load(f)

    n = args.n_per_class
    sl = slice(None) if n is not None and n < 0 else slice(0, n)
    harmful_sample = data.get("harmful", [])[sl]
    benign_sample = data.get("benign", [])[sl]

    prompts = [{"text": p, "label": "harmful"} for p in harmful_sample] + \
              [{"text": p, "label": "benign"} for p in benign_sample]

    # Apply sharding slicing
    if args.num_shards > 1:
        prompts = prompts[args.shard_id :: args.num_shards]

    mode = f"fixed target={args.target!r}" if args.target else "self-generated target"
    print(f"[INFO Shard {args.shard_id}/{args.num_shards}] Loaded {len(prompts)} prompts ({mode}). Output -> {args.out_file}")

    # Load model
    if args.family == "diffugpt":
        mm = ModelManager(family="diffugpt",
                          device_map=("cuda" if torch.cuda.is_available() else "cpu"),
                          torch_dtype=torch.float32)
    else:
        mm = ModelManager(family="dream", device_map="auto",
                          torch_dtype=torch.bfloat16)
                          
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()

    backend = build_backend(model, tokenizer, family=args.family)
    print(f"[INFO] Backend: {backend.family}  predicts_shifted={backend.predicts_shifted}")

    pad_id = tokenizer.pad_token_id or 0
    mask_token_id = backend.mask_token_id()

    n_layers = backend.num_layers()
    valid_layers = [l for l in args.layers if int(l) < n_layers]
    mlhm = MultiLayerHookManager(model, layer_specs=valid_layers, backend=backend)

    dlig = DLIGAttribution(
        model, tokenizer, mlhm.get_layer_view(valid_layers[0]),
        integration_steps=args.m, integration_batch_size=args.chunk,
        disable_kv_cache=True, score_mode="logprob",
        use_partial_forward=True,
        backend=backend,
    )
    
    if args.target:
        dlig.set_target_output(args.target)
    else:
        dlig.target_output_ids = None

    processed_prompts = set()
    if os.path.exists(args.out_file):
        with open(args.out_file, "r") as f:
            for line in f:
                if line.strip():
                    processed_prompts.add(json.loads(line)["prompt"])
        print(f"[INFO] Found {len(processed_prompts)} processed prompts. Resuming...")

    for item in tqdm(prompts, desc=f"Processing shard {args.shard_id}"):
        prompt = item["text"]
        label = item["label"]

        if prompt in processed_prompts:
            continue

        input_ids, attention_mask, L = build_prompt_inputs(
            tokenizer, args.system, prompt, device
        )

        keep_idx = input_token_indices(input_ids[0, :L].tolist(), tokenizer, user_prompt=prompt)
        prompt_tokens = [clean_token(t) for t in tokenizer.convert_ids_to_tokens(input_ids[0, :L])]
        kept_labels = [prompt_tokens[i] for i in keep_idx]

        rec = TrajRecorder()
        _ = backend.generate_trajectory(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens, steps=args.gen_steps,
            record_hook=rec.hook,
        )

        prompt_results = {
            "prompt": prompt, "label": label, "input_tokens": kept_labels, "steps_data": []
        }

        for step in args.target_steps:
            if step not in rec.x_by_step:
                continue

            x_t = rec.x_by_step[step].to(device)

            baseline_inp = x_t.clone()
            baseline_inp[:, :L] = mask_token_id

            with torch.no_grad():
                real_acts = mlhm.capture_activations(x_t, disable_kv_cache=True)
                baseline_acts = mlhm.capture_activations(baseline_inp, disable_kv_cache=True)

            step_data = {"step": step, "layers": {}}

            dlig.set_original_input_length(L)

            for layer in valid_layers:
                dlig.hook_manager = mlhm.get_layer_view(layer)

                res = dlig.compute_dlig_at_timestep_with_activations(
                    step=step, x_t=x_t,
                    real_act=real_acts[layer], baseline_act=baseline_acts[layer],
                    original_length=L,
                )

                full_dlig = res["full_dlig"][0].sum(dim=-1).float().numpy() 
                filtered_dlig = full_dlig[keep_idx].tolist()
                step_data["layers"][layer] = filtered_dlig

            prompt_results["steps_data"].append(step_data)

        with open(args.out_file, "a") as f:
            f.write(json.dumps(prompt_results) + "\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()