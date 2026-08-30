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
def infill_generate_trajectory(backend, x_full, span_slice, steps, record_hook=None):
    """
    x_full:   [1, L_total] = [left | span | right], span already MASK-filled.
    span_slice: (gap_start, gap_end) maskable middle region.
    steps:    number of diffusion steps T.
    record_hook: called each step with (step, xt_cpu, logits) for callers that
        need the trajectory (e.g. this file's own DLIG attribution loop below).
        Pass None (or omit) to skip the per-step .detach().to("cpu").clone()
        entirely -- a real cost (CUDA sync + host transfer) that a bare
        generate-and-score caller (scripts/eval_task.py) doesn't need, since it
        only wants the final x0.
    Returns final x0.
    """
    device = next(backend.lm_head.parameters()).device
    mask_id = backend.mask_token_id()
    gap_start, gap_end = span_slice

    x = x_full.to(device)
    # maskable = ONLY the middle span; both contexts are src.
    maskable_mask = torch.zeros_like(x, dtype=torch.bool)
    maskable_mask[:, gap_start:gap_end] = True

    # t = T : span fully masked (already is, but enforce)
    xt = x.masked_fill(maskable_mask, mask_id)

    def _predict(xt_in):
        vocab = backend.wte.num_embeddings
        if (xt_in >= vocab).any() or (xt_in < 0).any():
            raise ValueError(f"token id out of range for wte (vocab={vocab})")
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
        record_hook(0, xt.detach().to("cpu").clone(), logits)

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
            record_hook(rec_idx, xt.detach().to("cpu").clone(), logits)
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

    p.add_argument("--n_samples", type=int, default=1000,
                   help="paper uses first 1000 ROCStories cases.")
    p.add_argument("--max_side_tokens", type=int, default=120,
                   help="cap tokens kept per side (left/right) for the position axis.")

    # DLIG hyperparameters
    p.add_argument("--m", type=int, default=8, help="Integration steps.")
    p.add_argument("--chunk", type=int, default=12, help="Integration batch size.")
    p.add_argument("--gen_steps", type=int, default=12,
                   help="Diffusion denoising steps T for the infill span.")
    p.add_argument("--target_steps", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11],
                   help="Which recorded denoising steps to attribute at.")
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

    for sidx, sents in enumerate(tqdm(stories, desc=f"infill shard {args.shard_id}")):
        story_id = f"{args.shard_id}:{sidx}"
        if story_id in processed:
            continue

        s1, s2, s3, s4, s5 = sents

        # tokenize the three regions. Join with spaces; encode each region so we
        # know exact token boundaries. (No BOS/EOS injection; DiffuGPT is GPT-2
        # base, plain text.)
        left_ids = tokenizer.encode(" ".join([s1, s2]), return_tensors="pt").to(device)
        gold_ids = tokenizer.encode(" " + s3, return_tensors="pt").to(device)   # leading space: BPE-consistent
        right_ids = tokenizer.encode(" " + " ".join([s4, s5]), return_tensors="pt").to(device)

        # cap each side
        if left_ids.shape[1] > args.max_side_tokens:
            left_ids = left_ids[:, -args.max_side_tokens:]      # keep nearest-to-gap
        if right_ids.shape[1] > args.max_side_tokens:
            right_ids = right_ids[:, :args.max_side_tokens]      # keep nearest-to-gap

        span_len = gold_ids.shape[1]
        n_left = left_ids.shape[1]
        n_right = right_ids.shape[1]
        if span_len < 1 or n_left < 1 or n_right < 1:
            continue

        gap_start = n_left                      # first masked (target) position
        gap_end = n_left + span_len             # one past last target position
        L_total = n_left + span_len + n_right   # full sequence length

        mask_block = torch.full((1, span_len), mask_token_id,
                                dtype=left_ids.dtype, device=device)
        x_t = torch.cat([left_ids, mask_block, right_ids], dim=1)  # [1, L_total]

        # ----------------------------------------------------------------- #
        # Target framing.
        #   gold: fixed target = gold sentence 3 at the gap. The contrastive-
        #         target path scores logits[:, L : L+score_len] against
        #         target_output_ids with L == original_input_length = gap_start.
        #   self: target_output_ids = None -> the self-generated path scores the
        #         span tokens the model has committed at THIS step (masks
        #         excluded). score_window restricts scoring to the gap so the
        #         fixed right context is never scored.
        # Attribution from _process_results is sliced to relevant_token_indices,
        # which we set to the full context index set (both sides of the gap).
        # ----------------------------------------------------------------- #
        if args.target_mode == "gold":
            dlig.target_output_ids = gold_ids.squeeze(0)        # [span_len]
            dlig.score_window = None
        else:  # "self"
            dlig.target_output_ids = None
            dlig.score_window = (gap_start, gap_end)
        dlig.set_original_input_length(gap_start)

        # context positions to attribute over = everything EXCEPT the gap:
        ctx_indices = list(range(0, gap_start)) + list(range(gap_end, L_total))
        dlig.relevant_token_indices = ctx_indices

        # token strings + signed distance for each kept context position
        all_tok_strs = [clean_token(t) for t in
                        tokenizer.convert_ids_to_tokens(x_t[0])]
        ctx_tok_strs, ctx_signed_dist = [], []
        for i in ctx_indices:
            ctx_tok_strs.append(all_tok_strs[i])
            if i < gap_start:
                ctx_signed_dist.append(i - gap_start)          # negative (left)
            else:
                ctx_signed_dist.append(i - (gap_end - 1))      # positive (right)

        # ---- run the infill denoising trajectory (middle-masked span) ---- #
        from types import SimpleNamespace
        rec = SimpleNamespace(x_by_step={})
        def _rec_hook(step, xt_cpu, logits):
            rec.x_by_step[int(step)] = xt_cpu
        with torch.no_grad():
            final_x0 = infill_generate_trajectory(
                backend, x_t, (gap_start, gap_end),
                steps=args.gen_steps, record_hook=_rec_hook)

        # quality: ROUGE-1 of the FINAL generated span vs gold
        final_span = final_x0[0, gap_start:gap_end].cpu().tolist()
        rouge1 = rouge1_f1(final_span, gold_ids.squeeze(0).cpu().tolist())

        story_result = {
            "story_id": story_id,
            "label": "rocstories_infill",
            "target_mode": args.target_mode,
            "n_left": n_left, "n_right": n_right, "span_len": span_len,
            "gold_s3": s3,
            "rouge1": rouge1,
            "input_tokens": ctx_tok_strs,        # kept context tokens, in ctx_indices order
            "signed_dist": ctx_signed_dist,      # signed distance per kept token (<0 left, >0 right)
            "steps_data": [],
        }

        # ---- attribute at each requested denoising step ---- #
        # gold: target stays FIXED to gold s3 across steps.
        # self: F_t at each step scores the span tokens committed SO FAR (a
        #       moving quantity); n_committed is logged per step so downstream
        #       analysis can filter or weight early steps (n_committed=0 =>
        #       F=0 => zero attribution, drop those cells).
        for step in args.target_steps:
            if step not in rec.x_by_step:
                continue
            x_step = rec.x_by_step[step].to(device)
            n_committed = int((x_step[0, gap_start:gap_end] != mask_token_id).sum().item())

            # Degenerate-step guard (self mode): if no committed span token is
            # scoreable, F == 0 identically -> all gradients are exactly zero
            # -> DLIG is a zero tensor. Skip the integration entirely and mark
            # the step; downstream analysis drops these cells via n_scoreable.
            # Under predicts_shifted the first span position (gap_start) has no
            # in-slice logit and cannot be scored, so it is excluded from the
            # scoreable count.
            if args.target_mode == "self":
                score_lo = gap_start + 1 if backend.predicts_shifted else gap_start
                n_scoreable = int((x_step[0, score_lo:gap_end] != mask_token_id).sum().item())
            else:
                n_scoreable = span_len          # gold target: always scoreable
            if n_scoreable == 0:
                story_result["steps_data"].append(
                    {"step": step, "n_committed": n_committed,
                     "n_scoreable": 0, "skipped": True, "layers": {}})
                continue

            # baseline for THIS step: mask the full context, keep the span state
            # (revealed-so-far tokens + remaining masks) identical.
            baseline_step = x_step.clone()
            for i in ctx_indices:
                baseline_step[0, i] = mask_token_id

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
                # full_dlig sliced to relevant_token_indices (= ctx_indices), [1, |ctx|, H]
                pos_scores = res["full_dlig"][0].sum(dim=-1).float().cpu().numpy()
                step_data["layers"][layer] = pos_scores.tolist()

            story_result["steps_data"].append(step_data)

        with open(args.out_file, "a") as f:
            f.write(json.dumps(story_result) + "\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()