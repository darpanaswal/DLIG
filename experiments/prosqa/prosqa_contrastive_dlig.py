#!/usr/bin/env python3
# experiments/prosqa_contrastive_dlig.py
"""
prosqa_contrastive_dlig.py — per-example contrastive DLIG on ProsQA-finetuned
DiffuGPT-M.

Differs from contrastive_attribution.py in one essential way: the fixed target
is PER-EXAMPLE, and there are TWO of them,

    y+ = gold answer sentence          ("Bob is a shumpus.")
    y- = same sentence, wrong option   ("Bob is a gerpus.")

Since F_t is a sum of per-position log-probs and DLIG is linear in F_t,
    dDLIG = DLIG(y+) - DLIG(y-)
isolates the prompt's contribution to preferring the correct option. Both raw
tensors are stored so analysis can also flip the contrast for the fail group
(behavior-aligned direction = -dDLIG when the model chose y-).

Efficiency: activations (a, a') are captured ONCE per (example, step) for all
layers; only the integration (grad of F) is done twice, once per target.

Per prompt token, a span_id links the token to the structural label produced by
prosqa_graph_labels.py (edge / root / question, gold-path hop, model-chain
membership). Mapping is char-offset based (GPT2TokenizerFast offsets), computed
here because only the runner holds the tokenizer.

Output: resumable JSONL keyed by idx, one record per example:
    {idx, group, bucket, k_gold, gold_option, wrong_option,
     input_tokens, span_ids,             # aligned lists, len = n prompt tokens
     steps_data: [{step, layers: {l: {plus: [...], minus: [...]}}}]}
"""

import os
import gc
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from models.backends import build_backend
from utils.config import OUTPUT_DIR
from attribution.dlig_attribution import DLIGAttribution
from attribution.hook_manager import MultiLayerHookManager
from models.model_manager import ModelManager
from experiments.theorems.verify_completeness import (
    TrajRecorder, set_seed, build_prompt_inputs,
)
from experiments.contrastive.contrastive_attribution import clean_token, input_token_indices


def resolve_sep_token_id(tokenizer):
    """Token id for the '======' separator DiffuGPT-M ddm-sft training used
    between question and CoT/answer (see diffugpt/scripts/prosqa_to_diffusft.py /
    eval_prosqa.py: sep_token_id 50155). The checkpoint here is loaded as a plain
    GPT2TokenizerFast (models/model_manager.py), so fall back to encoding the
    literal string if tokenizer.sep_token_id isn't set."""
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if sep_id is not None:
        return sep_id
    ids = tokenizer("======", add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"'======' did not resolve to a single token id: {ids}")
    return ids[0]


def append_sep_token(tokenizer, input_ids, attention_mask, L):
    """Append the training-format '======' separator after the prompt.
    DiffuGPT-M ProsQA training data is '<bos> question ====== <CoT> ### <answer>
    <eos>'; every generation must reproduce it or the model is off-distribution."""
    sep_id = resolve_sep_token_id(tokenizer)
    sep = torch.tensor([[sep_id]], dtype=input_ids.dtype, device=input_ids.device)
    input_ids = torch.cat([input_ids, sep], dim=1)
    if attention_mask is not None:
        ones = torch.ones((attention_mask.shape[0], 1),
                           dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, ones], dim=1)
    return input_ids, attention_mask, L + 1


def wrong_target(gold: str, gold_option: str, wrong_option: str) -> str:
    """
    y- = gold sentence with the final concept swapped. Replacing only the LAST
    word occurrence keeps every other character of the target identical, so the
    contrast is exactly {gold_option vs wrong_option} at the answer position(s).
    """
    import re
    m = list(re.finditer(re.escape(gold_option), gold, flags=re.IGNORECASE))
    if not m:
        # fall back to canonical template
        return gold.rstrip(". ") + f" [{wrong_option}]"
    last = m[-1]
    return gold[:last.start()] + wrong_option + gold[last.end():]


def token_span_ids(prompt: str, spans, tokenizer):
    """
    span_id per prompt token (list[int], -1 = unlabeled). A token is assigned
    to the labeled span with maximal char overlap. Alignment contract: the ids
    align 1:1 with tokenizer.encode(prompt, add_special_tokens=False), which is
    the same token list input_token_indices() locates inside the full input.
    """
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    out = []
    for (ts, te) in offsets:
        best, best_ov = -1, 0
        for sp in spans:
            s, e = sp["span"]
            ov = max(0, min(te, e) - max(ts, s))
            if ov > best_ov:
                best, best_ov = sp["id"], ov
        out.append(best)
    return ids, out


def build_arg_parser():
    p = argparse.ArgumentParser(description="ProsQA contrastive DLIG (per-example y+/y-)")
    p.add_argument("--graph_labels", type=str, required=True,
                   help="jsonl from prosqa_graph_labels.py")
    p.add_argument("--out_file", type=str,
                   default=str(OUTPUT_DIR / "prosqa/prosqa_dlig.jsonl"))
    p.add_argument("--model_path", type=str, required=True,
                   help="Fine-tuned DiffuGPT-M checkpoint dir (e.g. models/diffugpt-m-prosqa)")
    p.add_argument("--groups", type=str, nargs="+", default=["success", "fail"],
                   help="Which bucket groups to attribute. off is dropped by default.")
    p.add_argument("--n_per_group", type=int, default=-1, help="-1 => all")
    p.add_argument("--system", type=str, default="",
                   help="ProsQA finetuning used raw question format => empty system.")

    # Sharding
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Examples generated together per call to "
                        "generate_trajectory (left-padded; the backend's "
                        "padding-aware attention mask + position ids make "
                        "this equivalent to generating each alone -- verified "
                        "in scripts/verify_batching.py). DLIG attribution "
                        "itself stays per-example; only generation is "
                        "batched. 1 = original unbatched behavior.")

    # DLIG hyperparameters (paper setting: T=12, steps {1,3,5,7,9,11}, even layers)
    p.add_argument("--m", type=int, default=12)
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--gen_steps", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--target_steps", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63],
                   help="Denoising steps to attribute at, every 2nd step from 1 to gen_steps-1 -- same absolute cadence as the paper's T=12 setting ([1,3,5,7,9,11]), extended to T=64 (32 points). More points = more sequential attribution compute per example, NOT more peak memory (batch_size only affects the generation phase); override to match a different --gen_steps.")
    p.add_argument("--layers", type=str, nargs="+",
                   default=[str(i) for i in range(0, 24, 2)])
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    if args.num_shards > 1:
        op = Path(args.out_file)
        combined_file = args.out_file          # merged output from prior runs
        args.out_file = str(op.parent / f"{op.stem}_shard{args.shard_id}{op.suffix}")
    else:
        combined_file = None
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    # ensure the shard file exists even if 0 examples match, so a downstream
    # `cat *_shard*.jsonl` doesn't fail with "No such file or directory"
    Path(args.out_file).touch(exist_ok=True)

    # ---- load graph-labeled examples, filter groups, cap per group ----
    examples = []
    per_group = {}
    with open(args.graph_labels) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            g = rec["group"]
            if g not in args.groups:
                continue
            if args.n_per_group >= 0 and per_group.get(g, 0) >= args.n_per_group:
                continue
            per_group[g] = per_group.get(g, 0) + 1
            examples.append(rec)

    if not per_group:
        raise SystemExit(
            f"[ERROR] 0 examples matched --groups {args.groups} in "
            f"{args.graph_labels}. Check the group filter argument."
        )

    if args.num_shards > 1:
        examples = examples[args.shard_id :: args.num_shards]
    print(f"[INFO Shard {args.shard_id}/{args.num_shards}] {len(examples)} examples "
          f"(groups: {per_group}). Output -> {args.out_file}")

    # ---- model: finetuned DiffuGPT-M, float32 (matches contrastive pipeline) ----
    mm = ModelManager(family="diffugpt",
                      device_map=("cuda" if torch.cuda.is_available() else "cpu"),
                      torch_dtype=torch.float32,
                      model_path=args.model_path)
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()

    backend = build_backend(model, tokenizer, family="diffugpt")
    print(f"[INFO] Backend: {backend.family}  predicts_shifted={backend.predicts_shifted}")

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

    # ---- resume ----
    done = set()
    for rf in filter(None, [args.out_file, combined_file]):
        if os.path.exists(rf):
            with open(rf) as f:
                for line in f:
                    if line.strip():
                        done.add(json.loads(line)["idx"])
    if done:
        print(f"[INFO] Resuming past {len(done)} examples.")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    todo = [ex for ex in examples if ex["idx"] not in done]
    pbar = tqdm(total=len(todo), desc=f"shard {args.shard_id}")

    for chunk_start in range(0, len(todo), args.batch_size):
        chunk = todo[chunk_start: chunk_start + args.batch_size]

        # --- per-example prep (cheap, CPU-bound): prompt encoding, keep_idx/
        # span_ids computed on the PROMPT-ONLY encoding (before the separator
        # is appended), exactly as the original unbatched code did. ---
        prepped = []
        for ex in chunk:
            prompt = ex["question"]
            y_plus = ex["gold"]
            y_minus = wrong_target(ex["gold"], ex["gold_option"], ex["wrong_option"])

            input_ids, attention_mask, L = build_prompt_inputs(
                tokenizer, args.system, prompt, device
            )
            keep_idx = input_token_indices(input_ids[0, :L].tolist(), tokenizer,
                                           user_prompt=prompt)
            prompt_tokens = [clean_token(t)
                             for t in tokenizer.convert_ids_to_tokens(input_ids[0, :L])]
            kept_tokens = [prompt_tokens[i] for i in keep_idx]

            p_ids, span_ids = token_span_ids(prompt, ex["spans"], tokenizer)
            if len(span_ids) != len(keep_idx):
                print(f"[WARN idx={ex['idx']}] span/token misalign "
                      f"({len(span_ids)} vs {len(keep_idx)}); trimming.")
                span_ids = (span_ids + [-1] * len(keep_idx))[: len(keep_idx)]

            # append the separator AFTER keep_idx/span_ids (which index into
            # the prompt-only encoding); everything downstream uses this L.
            input_ids, attention_mask, L = append_sep_token(
                tokenizer, input_ids, attention_mask, L
            )
            prepped.append(dict(ex=ex, y_plus=y_plus, y_minus=y_minus,
                                keep_idx=keep_idx, kept_tokens=kept_tokens,
                                span_ids=span_ids, input_ids=input_ids,
                                attention_mask=attention_mask, L=L))

        # --- left-pad into one batch, generate the WHOLE BATCH's trajectory
        # in one call. The backend's padding-aware attention mask + position
        # ids make this equivalent to generating each example alone --
        # verified in scripts/verify_batching.py. ---
        Lmax = max(p["L"] for p in prepped)
        ids_rows, mask_rows = [], []
        for p in prepped:
            n_pad = Lmax - p["L"]
            pad_ids = torch.full((1, n_pad), pad_id, dtype=p["input_ids"].dtype, device=device)
            pad_am = torch.zeros((1, n_pad), dtype=p["attention_mask"].dtype, device=device)
            ids_rows.append(torch.cat([pad_ids, p["input_ids"]], dim=1))
            mask_rows.append(torch.cat([pad_am, p["attention_mask"]], dim=1))
        batch_ids = torch.cat(ids_rows, dim=0)
        batch_mask = torch.cat(mask_rows, dim=0)

        batch_rec = TrajRecorder()
        _ = backend.generate_trajectory(
            batch_ids, attention_mask=batch_mask,
            max_new_tokens=args.max_new_tokens, steps=args.gen_steps,
            record_hook=batch_rec.hook,
        )

        # --- per-example DLIG attribution: identical to the unbatched path,
        # just fed a slice of the batch (padding stripped) instead of a
        # freshly generated single-example tensor. ---
        for i, p in enumerate(prepped):
            ex = p["ex"]
            L = p["L"]
            keep_idx = p["keep_idx"]
            n_pad = Lmax - L

            out = {
                "idx": ex["idx"], "group": ex["group"], "bucket": ex["bucket"],
                "k_gold": ex["k_gold"], "k_wrong": ex["k_wrong"],
                "n_edges": ex["n_edges"],
                "gold_option": ex["gold_option"], "wrong_option": ex["wrong_option"],
                "y_plus": p["y_plus"], "y_minus": p["y_minus"],
                "input_tokens": p["kept_tokens"], "span_ids": p["span_ids"],
                "steps_data": [],
            }

            for step in args.target_steps:
                if step not in batch_rec.x_by_step:
                    continue
                x_t = batch_rec.x_by_step[step][i: i + 1, n_pad:].to(device)

                baseline_inp = x_t.clone()
                baseline_inp[:, :L] = mask_token_id

                # activations shared across layers AND across both targets:
                # a, a' depend only on (x_t, prompt mask), not on the readout y.
                with torch.no_grad():
                    real_acts = mlhm.capture_activations(x_t, disable_kv_cache=True)
                    baseline_acts = mlhm.capture_activations(baseline_inp,
                                                             disable_kv_cache=True)

                step_data = {"step": step, "layers": {}}
                dlig.set_original_input_length(L)

                for tgt_name, tgt in (("plus", p["y_plus"]), ("minus", p["y_minus"])):
                    dlig.set_target_output(tgt)
                    for layer in valid_layers:
                        dlig.hook_manager = mlhm.get_layer_view(layer)
                        res = dlig.compute_dlig_at_timestep_with_activations(
                            step=step, x_t=x_t,
                            real_act=real_acts[layer],
                            baseline_act=baseline_acts[layer],
                            original_length=L,
                        )
                        # per-position score: s[i] = sum_j DLIG[i, j]
                        s = res["full_dlig"][0].sum(dim=-1).float().numpy()
                        step_data["layers"].setdefault(layer, {})[tgt_name] = \
                            s[keep_idx].tolist()

                out["steps_data"].append(step_data)

            with open(args.out_file, "a") as f:
                f.write(json.dumps(out) + "\n")
            pbar.update(1)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pbar.close()


if __name__ == "__main__":
    main()