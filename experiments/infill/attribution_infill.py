# experiments/attribution_infill.py
"""
ROCStories sentence-infilling DLIG attribution (the DLM-native task).

WHY THIS TASK
-------------
DiffuGPT (Lou et al., ICLR 2025) evaluates story infilling on ROCStories:
each case is a 5-sentence story; the model infills SENTENCE 3 conditioned on
sentences 1-2 (LEFT context) and 4-5 (RIGHT context), scored by ROUGE-1/2/L.
This is the setting where DLMs are structurally distinct from AR models: the
infilled span attends to BOTH sides (bidirectional), the span is multi-token,
and it denoises over real timesteps (paper uses T=64).

This is the right task for DLIG because every lever is active:
  - multi-token target  -> |y_t| >> 1 (the parallel-denoising structure DLIG is for)
  - genuine timesteps   -> the layer x timestep trajectory is meaningful
  - bidirectional long-range -> position axis is SIGNED (left vs right context),
    giving the clean diffusion-vs-AR contrast (AR can only see the left side).

DESIGN
------
  context = [s1 s2]  <gap = MASK x span_len>  [s4 s5]
  target  = gold sentence 3 (fixed via target_output_ids), scored at the gap.
  attribution -> ALL context positions (both sides).
  position axis = SIGNED distance to the span:
      left-context token at index i  (i < gap_start): dist = i - gap_start  (<0)
      right-context token at index i  (i >= gap_end):  dist = i - (gap_end-1) (>0)

CORRECTNESS / QUALITY
---------------------
Per story we log ROUGE-1 of the model's argmax infill vs gold (cheap, no extra
deps), enabling a quality split (high- vs low-ROUGE).

SHIFT NOTE (Dream/LLaDA swap)
-----------------------------
The fixed-target scoring path in DLIGAttribution does NOT apply the predicts_shifted
offset (only the self-generated path does). Correct for DiffuGPT (predicts_shifted
=False). When swapping to a shifted backend, the gap must be aligned to the shift
(see _compute_target_score). The correctness check below DOES branch on the shift.
"""

import os
import gc
import json
import argparse
from pathlib import Path

import torch
import torch.distributions as dists
from tqdm import tqdm

from utils.config import OUTPUT_DIR
from models.model_manager import ModelManager
from models.backends import build_backend
from models.backends.diffugpt import top_p_logits   # nucleus filter used by the sampler
from attribution.hook_manager import MultiLayerHookManager
from attribution.dlig_attribution import DLIGAttribution


def clean_token(tok: str) -> str:
    # mirrors experiments/attribution.py
    return tok.replace("\u0120", "_").replace("\u2581", "_").replace("\n", "\\n").replace("\t", "\\t")


# --------------------------------------------------------------------------- #
#  ROUGE-1 (unigram F1) — dependency-free, for the quality split
# --------------------------------------------------------------------------- #
def rouge1_f1(pred_ids, gold_ids):
    """Token-level unigram overlap F1 between two id lists (multiset)."""
    from collections import Counter
    p, g = Counter(pred_ids), Counter(gold_ids)
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    prec = overlap / max(1, sum(p.values()))
    rec = overlap / max(1, sum(g.values()))
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- #
#  Infill denoising trajectory (MIDDLE-masked span)
# --------------------------------------------------------------------------- #
#  backend.generate_trajectory masks the SUFFIX (prompt = src at [:, :L]). For
#  infilling the masked span is in the MIDDLE: both left AND right context are
#  src (never re-masked), only the span denoises. We mirror the HKUNLP random-
#  reveal sampler (same p_to_x0 = 1/(t+1) schedule, top-p filtering, shift
#  handling) but with the maskable region = the middle span only.
#
#  Records xt at each step via record_hook(step, xt, logits), exactly like
#  TrajRecorder, so target_steps line up with generation step counts.
# --------------------------------------------------------------------------- #
def infill_generate_trajectory(backend, x_full, span_slice, steps, record_hook=None,
                                key_padding_mask=None, position_ids=None):
    """
    x_full:   [B, L_total]. Single-example (B=1): [left | span | right], span
        already MASK-filled. Batched (B>1): each row laid out
        [left(row) | span(row) | right(row) | trailing_pad(row)], left-aligned
        (no leading padding, unlike generate_trajectory's prefix+trailing
        layout) since infill's span position varies per row -- trailing pad
        is the only padding needed once maskable_mask/key_padding_mask/
        position_ids are per-row (see run_infill_batch below).
    span_slice: EITHER a single (gap_start, gap_end) tuple, broadcast to every
        row (the original, single-example usage -- unchanged), OR a full
        [B, L_total] bool tensor for batched calls where each row's span sits
        at a different position/length.
    steps:    number of diffusion steps T.
    record_hook: called each step with (step, xt_cpu, logits) for callers that
        need the trajectory (e.g. this file's own DLIG attribution loop below).
        Pass None (or omit) to skip the per-step .detach().to("cpu").clone()
        entirely -- a real cost (CUDA sync + host transfer) that a bare
        generate-and-score caller (helpers/eval_task.py) doesn't need, since it
        only wants the final x0.
    key_padding_mask/position_ids: see models/backends/diffugpt.py's
        forward_logits. None (default) reproduces the original single-example
        behavior exactly (every existing caller omits them); required
        together for correct batched generation with trailing padding.
    Returns final x0.
    """
    device = next(backend.lm_head.parameters()).device
    mask_id = backend.mask_token_id()

    x = x_full.to(device)
    if isinstance(span_slice, tuple):
        gap_start, gap_end = span_slice
        maskable_mask = torch.zeros_like(x, dtype=torch.bool)
        maskable_mask[:, gap_start:gap_end] = True
    else:
        maskable_mask = span_slice.to(device=device, dtype=torch.bool)

    # t = T : span fully masked (already is, but enforce)
    xt = x.masked_fill(maskable_mask, mask_id)

    def _predict(xt_in):
        vocab = backend.wte.num_embeddings
        if (xt_in >= vocab).any() or (xt_in < 0).any():
            raise ValueError(f"token id out of range for wte (vocab={vocab})")
        # key_padding_mask/position_ids are DiffuGPT-only kwargs (dream.py's
        # DreamBackend.forward_logits doesn't accept them at all); only pass
        # them through when actually needed, so Dream (--family dream) is
        # unaffected regardless of these being None.
        if key_padding_mask is not None or position_ids is not None:
            logits = backend.forward_logits(xt_in, key_padding_mask=key_padding_mask,
                                            position_ids=position_ids)
        else:
            logits = backend.forward_logits(xt_in)
        filt = top_p_logits(logits / backend.logits_temp, p=backend.topp_temp)
        scores = torch.log_softmax(filt, dim=-1)
        x0 = dists.Categorical(logits=scores).sample()
        if backend.predicts_shifted:
            x0 = torch.cat([x[:, 0:1], x0[:, :-1]], dim=1)   # shift right by one
        # keep already-revealed (non-maskable) positions identical to xt
        x0 = xt_in.masked_scatter(maskable_mask, x0[maskable_mask])
        return logits, x0

    # step index 0 (t = T): span fully masked
    logits, x0 = _predict(xt)
    if record_hook is not None:
        record_hook(0, xt.detach().clone(), logits)

    # steps t = T-1 .. 1: progressively reveal span tokens
    rec_idx = 1
    cur_maskable = maskable_mask.clone()
    for t in range(steps - 1, 0, -1):
        p_to_x0 = 1.0 / (t + 1)
        reveal = cur_maskable & (torch.rand_like(x0, dtype=torch.float) < p_to_x0)
        xt = xt.masked_scatter(reveal, x0[reveal])
        cur_maskable = cur_maskable.masked_fill(reveal, False)
        logits, x0 = _predict(xt)
        if record_hook is not None:
            record_hook(rec_idx, xt.detach().clone(), logits)
        rec_idx += 1

    return x0


# --------------------------------------------------------------------------- #
#  ROCStories loading: 5 sentences per story
# --------------------------------------------------------------------------- #
def load_stories(dataset_arg, n_samples):
    """Return list of [s1,s2,s3,s4,s5]. Accepts a local JSONL (one story per
    line, either {'sentences':[...]} or {'sentence1':..,..,'sentence5':..}) or
    an HF dataset id."""
    stories = []
    if os.path.exists(str(dataset_arg)):
        with open(str(dataset_arg), "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "sentences" in row and len(row["sentences"]) == 5:
                    stories.append([s.strip() for s in row["sentences"]])
                elif all(f"sentence{i}" in row for i in range(1, 6)):
                    stories.append([row[f"sentence{i}"].strip() for i in range(1, 6)])
    else:
        from datasets import load_dataset
        ds = load_dataset(str(dataset_arg), split="test")
        for row in ds:
            if all(f"sentence{i}" in row for i in range(1, 6)):
                stories.append([row[f"sentence{i}"].strip() for i in range(1, 6)])
            elif "sentences" in row and len(row["sentences"]) == 5:
                stories.append([s.strip() for s in row["sentences"]])

    # paper uses the first 1000 for efficiency
    if n_samples > 0:
        stories = stories[:n_samples]
    return stories


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="ROCStories infilling DLIG attribution over (layer, [timestep])")
    p.add_argument("--family", type=str, default="diffugpt",
                   choices=["dream", "diffugpt"])
    # local JSONL path OR an HF dataset id (e.g. 'Ximing/ROCStories')
    p.add_argument("--dataset", type=str,
                   default=str(OUTPUT_DIR.parent / "data/rocstories_test.jsonl"))
    p.add_argument("--out_file", type=str,
                   default=str(OUTPUT_DIR / "infill_attribution/rocstories_infill_attribution.jsonl"))

    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Stories generated together per call to "
                        "infill_generate_trajectory (left|span|right|"
                        "trailing_pad layout; verified via "
                        "helpers/verify_batching.py --task infill -- requires "
                        "TF32 disabled, see model_manager.py). DLIG "
                        "attribution stays per-story; only generation is "
                        "batched. diffugpt only (--family dream ignores this). "
                        "1 = original unbatched behavior.")

    p.add_argument("--n_samples", type=int, default=1000,
                   help="paper uses first 1000 ROCStories cases.")
    p.add_argument("--max_side_tokens", type=int, default=120,
                   help="cap tokens kept per side (left/right) for the position axis.")

    # DLIG hyperparameters
    p.add_argument("--m", type=int, default=8, help="Integration steps.")
    p.add_argument("--chunk", type=int, default=12, help="Integration batch size.")
    p.add_argument("--gen_steps", type=int, default=64,
                   help="Diffusion denoising steps T for the infill span.")
    p.add_argument("--target_steps", type=int, nargs="+",
                   default=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
                            31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57,
                            59, 61, 63],
                   help="Denoising steps to attribute at, every 2nd step from 1 "
                        "to gen_steps-1 -- same absolute cadence as the paper's "
                        "T=12 setting ([1,3,5,7,9,11]), extended to T=64 (32 "
                        "points). More points = more sequential attribution "
                        "compute per story, NOT more peak memory (batch_size "
                        "only affects the generation phase); override to match "
                        "a different --gen_steps.")
    p.add_argument("--target_mode", type=str, default="self",
                   choices=["self", "gold"],
                   help="'self' (primary): F_t scores the model's own committed span "
                        "tokens at each step (masks excluded), matching the method's "
                        "default framing; no oracle completion enters the score. "
                        "'gold': fixed target = gold sentence 3 held constant across "
                        "steps (robustness / appendix design).")
    p.add_argument("--layers", type=str, nargs="+",
                   default=[str(i) for i in range(0, 26, 2)])
    p.add_argument("--score_mode", type=str, default="meancentered",
                   choices=["meancentered", "logprob"])
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)

    if args.num_shards > 1:
        out_path = Path(args.out_file)
        args.out_file = str(out_path.parent /
                            f"{out_path.stem}_shard{args.shard_id}{out_path.suffix}")
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    stories = load_stories(args.dataset, args.n_samples)
    if args.num_shards > 1:
        stories = stories[args.shard_id::args.num_shards]
    print(f"[INFO Shard {args.shard_id}/{args.num_shards}] "
          f"{len(stories)} ROCStories. Output -> {args.out_file}")

    # ---- model ----
    if args.family == "diffugpt":
        mm = ModelManager(family="diffugpt",
                          device_map="cuda" if torch.cuda.is_available() else "cpu",
                          torch_dtype=torch.float32)
    else:
        mm = ModelManager(family="dream", device_map="auto",
                          torch_dtype=torch.bfloat16)
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()
    backend = build_backend(model, tokenizer, family=args.family)

    mask_token_id = backend.mask_token_id()
    n_layers = backend.num_layers()
    valid_layers = [l for l in args.layers if int(l) < n_layers]
    mlhm = MultiLayerHookManager(model, layer_specs=valid_layers, backend=backend)

    dlig = DLIGAttribution(
        model, tokenizer, mlhm.get_layer_view(valid_layers[0]),
        integration_steps=args.m, integration_batch_size=args.chunk,
        disable_kv_cache=True, score_mode=args.score_mode,
        use_partial_forward=True, backend=backend,
    )

    # ---- resume ----
    # Check both this shard's file AND the merged combined file: after the .sh
    # consolidation phase the shard files are deleted, so a restart must not
    # recompute stories that already live in the merged output.
    processed = set()
    resume_files = [args.out_file]
    if args.num_shards > 1:
        resume_files.append(str(Path(args.out_file).parent /
                                f"{Path(args.out_file).stem.rsplit('_shard', 1)[0]}"
                                f"{Path(args.out_file).suffix}"))
    for rf in resume_files:
        if os.path.exists(rf):
            with open(rf, "r") as f:
                for line in f:
                    if line.strip():
                        processed.add(json.loads(line)["story_id"])
    if processed:
        print(f"[INFO] Resuming; {len(processed)} stories already done "
              f"(checked: {resume_files}).")

    batch_size = args.batch_size if args.family == "diffugpt" else 1
    todo = [(sidx, sents) for sidx, sents in enumerate(stories)
            if f"{args.shard_id}:{sidx}" not in processed]
    pbar = tqdm(total=len(todo), desc=f"infill shard {args.shard_id}")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for chunk_start in range(0, len(todo), batch_size):
        chunk = todo[chunk_start: chunk_start + batch_size]

        # --- per-story encode (CPU-bound): tokenize + side-cap, same as the
        # original unbatched code. ---
        prepped = []
        for sidx, sents in chunk:
            s1, s2, s3, s4, s5 = sents
            left_ids = tokenizer.encode(" ".join([s1, s2]), return_tensors="pt")
            gold_ids = tokenizer.encode(" " + s3, return_tensors="pt")
            right_ids = tokenizer.encode(" " + " ".join([s4, s5]), return_tensors="pt")
            if left_ids.shape[1] > args.max_side_tokens:
                left_ids = left_ids[:, -args.max_side_tokens:]
            if right_ids.shape[1] > args.max_side_tokens:
                right_ids = right_ids[:, :args.max_side_tokens]
            span_len, n_left, n_right = gold_ids.shape[1], left_ids.shape[1], right_ids.shape[1]
            if span_len < 1 or n_left < 1 or n_right < 1:
                pbar.update(1)
                continue
            prepped.append(dict(sidx=sidx, s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
                                left_ids=left_ids, gold_ids=gold_ids, right_ids=right_ids,
                                n_left=n_left, n_right=n_right, span_len=span_len))
        if not prepped:
            continue

        # --- left|span|right|trailing_pad batch (left-aligned; span position
        # AND length vary per story, unlike wic/prosqa's shared prefix
        # layout -- generalized key_padding_mask/position_ids in
        # models/backends/diffugpt.py make this equivalent to generating each
        # story alone, verified in helpers/verify_batching.py). ---
        real_lens = [p["n_left"] + p["span_len"] + p["n_right"] for p in prepped]
        Smax = max(real_lens)
        B = len(prepped)
        x = torch.full((B, Smax), pad_id, dtype=torch.long)
        kpm = torch.zeros((B, Smax), dtype=torch.long)
        maskable = torch.zeros((B, Smax), dtype=torch.bool)
        for i, p in enumerate(prepped):
            gap_start, gap_end = p["n_left"], p["n_left"] + p["span_len"]
            real_len = real_lens[i]
            x[i, :p["n_left"]] = p["left_ids"][0]
            x[i, gap_start:gap_end] = mask_token_id
            x[i, gap_end:real_len] = p["right_ids"][0]
            kpm[i, :real_len] = 1
            maskable[i, gap_start:gap_end] = True
        x = x.to(device); kpm = kpm.to(device); maskable = maskable.to(device)
        position_ids = (kpm.cumsum(dim=1) - 1).clamp(min=0)

        from types import SimpleNamespace
        batch_rec = SimpleNamespace(x_by_step={})
        def _rec_hook(step, xt_cpu, logits):
            batch_rec.x_by_step[int(step)] = xt_cpu
        # key_padding_mask/position_ids are DiffuGPT-only (DreamBackend.
        # forward_logits doesn't accept them at all); only pass them for
        # diffugpt, so --family dream (always batch_size=1 here) is
        # unaffected even though kpm/position_ids were still computed above.
        gen_kwargs = dict(key_padding_mask=kpm, position_ids=position_ids) \
            if args.family == "diffugpt" else {}
        with torch.no_grad():
            final_x0_batch = infill_generate_trajectory(
                backend, x, maskable, steps=args.gen_steps, record_hook=_rec_hook,
                **gen_kwargs)

        # --- per-story post-processing: identical to the unbatched path,
        # just fed a slice of the batch (trailing padding stripped) instead
        # of a freshly generated single-story tensor. ---
        for i, p in enumerate(prepped):
            story_id = f"{args.shard_id}:{p['sidx']}"
            s3 = p["s3"]
            gold_ids = p["gold_ids"].to(device)
            n_left, n_right, span_len = p["n_left"], p["n_right"], p["span_len"]
            gap_start, gap_end = n_left, n_left + span_len
            L_total = real_lens[i]

            final_x0 = final_x0_batch[i: i + 1, :L_total]
            x_t = x[i: i + 1, :L_total]

            if args.target_mode == "gold":
                dlig.target_output_ids = gold_ids.squeeze(0)
                dlig.score_window = None
            else:
                dlig.target_output_ids = None
                dlig.score_window = (gap_start, gap_end)
            dlig.set_original_input_length(gap_start)

            ctx_indices = list(range(0, gap_start)) + list(range(gap_end, L_total))
            dlig.relevant_token_indices = ctx_indices

            all_tok_strs = [clean_token(t) for t in
                            tokenizer.convert_ids_to_tokens(x_t[0])]
            ctx_tok_strs, ctx_signed_dist = [], []
            for j in ctx_indices:
                ctx_tok_strs.append(all_tok_strs[j])
                if j < gap_start:
                    ctx_signed_dist.append(j - gap_start)
                else:
                    ctx_signed_dist.append(j - (gap_end - 1))

            final_span = final_x0[0, gap_start:gap_end].cpu().tolist()
            rouge1 = rouge1_f1(final_span, gold_ids.squeeze(0).cpu().tolist())

            story_result = {
                "story_id": story_id,
                "label": "rocstories_infill",
                "target_mode": args.target_mode,
                "n_left": n_left, "n_right": n_right, "span_len": span_len,
                "gold_s3": s3,
                "rouge1": rouge1,
                "input_tokens": ctx_tok_strs,
                "signed_dist": ctx_signed_dist,
                "steps_data": [],
            }

            for step in args.target_steps:
                if step not in batch_rec.x_by_step:
                    continue
                x_step = batch_rec.x_by_step[step][i: i + 1, :L_total].to(device)
                n_committed = int((x_step[0, gap_start:gap_end] != mask_token_id).sum().item())

                if args.target_mode == "self":
                    score_lo = gap_start + 1 if backend.predicts_shifted else gap_start
                    n_scoreable = int((x_step[0, score_lo:gap_end] != mask_token_id).sum().item())
                else:
                    n_scoreable = span_len
                if n_scoreable == 0:
                    story_result["steps_data"].append(
                        {"step": step, "n_committed": n_committed,
                         "n_scoreable": 0, "skipped": True, "layers": {}})
                    continue

                baseline_step = x_step.clone()
                for j in ctx_indices:
                    baseline_step[0, j] = mask_token_id

                with torch.no_grad():
                    real_acts = mlhm.capture_activations(x_step, disable_kv_cache=True)
                    baseline_acts = mlhm.capture_activations(baseline_step, disable_kv_cache=True)

                step_data = {"step": step, "n_committed": n_committed,
                             "n_scoreable": n_scoreable, "layers": {}}
                for layer in valid_layers:
                    dlig.hook_manager = mlhm.get_layer_view(layer)
                    res = dlig.compute_dlig_at_timestep_with_activations(
                        step=step, x_t=x_step,
                        real_act=real_acts[layer], baseline_act=baseline_acts[layer],
                        original_length=gap_start,
                    )
                    pos_scores = res["full_dlig"][0].sum(dim=-1).float().cpu().numpy()
                    step_data["layers"][layer] = pos_scores.tolist()

                story_result["steps_data"].append(step_data)

            with open(args.out_file, "a") as f:
                f.write(json.dumps(story_result) + "\n")
            pbar.update(1)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pbar.close()


if __name__ == "__main__":
    main()