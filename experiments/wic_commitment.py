#!/usr/bin/env python3
# experiments/wic_commitment.py
"""
wic_commitment.py — when does the answer commit along the DENOISING axis?

Diffusion analogue of the CoT-faithfulness "decided before it reasoned" probe.
An AR model has no position before the question (it's at the end of the prompt),
but a diffusion LM refines the answer over T denoising steps — that IS the time
axis. We read P(Yes) at the answer slot at every denoising step and ask WHEN the
model commits.

Hypothesis (population-level, not per-example): a bias-defaulted Yes locks in
EARLY, before later steps integrate the disambiguating context; a genuine answer
crystallises LATER. Pairs with the DLIG depth finding: biased Yes is shallow in
LAYER and early in DENOISING TIME — same shortcut, two axes.

Reuses generate_trajectory (same sampling as wic.py); NO DLIG, NO baseline acts,
so it's a handful of forward passes per example — much cheaper than the attribution
run. The hook keeps the logits generate_trajectory already computes and passes in.

Shift convention: backend.predicts_shifted => the readout for canvas position i
is at logits index i-1. We locate the answer slot from the committed x0 (the token
that decodes to Yes/No) and read its logits at slot-1.

Output JSONL, one record per example:
    {idx, word, label, pred, correct, gen_text, answer_slot,
     steps: [s0, s1, ...],           # recorded denoising step indices
     p_yes: [...], p_no: [...],      # raw softmax prob at the answer token id
     p_yes_2way: [...]}              # P(Yes)/(P(Yes)+P(No)), renormalised
"""

import os
import gc
import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.backends import build_backend
from utils.config import OUTPUT_DIR
from models.model_manager import ModelManager
from experiments.theorems.verify_completeness import set_seed, build_prompt_inputs
from experiments.wic import wic_prompt, read_pred, load_wic


def resolve_yes_no_ids(tokenizer):
    """Token ids for the answer tokens as they appear in '### Yes.' / '### No.'.
    Training target is '### Yes.<|endoftext|>', so the discriminating token is the
    space-prefixed 'Yes'/'No' (GPT-2 BPE 'ĠYes'/'ĠNo'). Resolve by tokenizing the
    exact target and taking the token whose decode strips to yes/no."""
    def pick(target, wanted):
        ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        for tid in ids:
            if tokenizer.decode([tid]).strip().lower() == wanted:
                return tid
        raise ValueError(f"could not find {wanted!r} token in {target!r} -> {ids}")
    return pick("### Yes.", "yes"), pick("### No.", "no")


def build_arg_parser():
    p = argparse.ArgumentParser(description="WiC denoising-step commitment probe")
    p.add_argument("--wic_jsonl", type=str, default="data/wic_test_raw.jsonl")
    p.add_argument("--out_file", type=str,
                   default=str(OUTPUT_DIR / "wic/wic_commitment.jsonl"))
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--n", type=int, default=-1)
    p.add_argument("--system", type=str, default="")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    # keep generation identical to wic.py so trajectories match
    p.add_argument("--gen_steps", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    if args.num_shards > 1:
        op = Path(args.out_file)
        combined_file = args.out_file
        args.out_file = str(op.parent / f"{op.stem}_shard{args.shard_id}{op.suffix}")
    else:
        combined_file = None
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    Path(args.out_file).touch(exist_ok=True)

    rows = load_wic(args.wic_jsonl)
    if args.n >= 0:
        rows = rows[:args.n]

    mm = ModelManager(family="diffugpt",
                      device_map=("cuda" if torch.cuda.is_available() else "cpu"),
                      torch_dtype=torch.float32,
                      model_path=args.model_path)
    model, tokenizer = mm.load_model_and_tokenizer()
    device = mm.get_model_device()

    backend = build_backend(model, tokenizer, family="diffugpt")
    shifted = backend.predicts_shifted
    yes_id, no_id = resolve_yes_no_ids(tokenizer)
    print(f"[INFO] predicts_shifted={shifted}  yes_id={yes_id} no_id={no_id}")

    indexed = list(enumerate(rows))
    if args.num_shards > 1:
        indexed = indexed[args.shard_id :: args.num_shards]

    done = set()
    for rf in filter(None, [args.out_file, combined_file]):
        if os.path.exists(rf):
            with open(rf) as f:
                for line in f:
                    if line.strip():
                        done.add(json.loads(line)["idx"])
    if done:
        print(f"[INFO] Resuming past {len(done)} examples.")

    for idx, r in tqdm(indexed, desc=f"shard {args.shard_id}"):
        if idx in done:
            continue

        prompt = wic_prompt(r["sentence1"], r["sentence2"], r["word"])
        input_ids, attention_mask, L = build_prompt_inputs(
            tokenizer, args.system, prompt, device
        )

        # capture logits at every denoising step
        rec = SimpleNamespace(logits_by_step={})
        def _hook(step, xt_cpu, logits):
            rec.logits_by_step[int(step)] = logits.detach().to("cpu")

        x0 = backend.generate_trajectory(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens, steps=args.gen_steps,
            record_hook=_hook,
        )

        gen_ids = x0[0][L:].tolist()
        eos_id = tokenizer.eos_token_id
        gen_ids_trim = gen_ids[:gen_ids.index(eos_id)] if (eos_id in gen_ids) else gen_ids
        gen_text = tokenizer.decode(gen_ids_trim, skip_special_tokens=True).strip()
        pred = read_pred(gen_text)
        correct = (pred is not None and pred == r["label"])

        # locate the answer slot: first canvas position >= L whose committed token
        # is yes_id or no_id. Fall back to L if neither committed (pred=None cases).
        answer_slot = None
        for j, tid in enumerate(gen_ids):
            if tid in (yes_id, no_id):
                answer_slot = L + j
                break
        if answer_slot is None:
            answer_slot = L  # unparseable; still record the curve at first gen slot

        read_pos = answer_slot - 1 if shifted else answer_slot

        steps = sorted(rec.logits_by_step)
        p_yes, p_no, p_yes_2way = [], [], []
        for s in steps:
            lg = rec.logits_by_step[s][0, read_pos]        # [|V|]
            probs = F.softmax(lg.float(), dim=-1)
            py = float(probs[yes_id]); pn = float(probs[no_id])
            p_yes.append(py); p_no.append(pn)
            p_yes_2way.append(py / (py + pn + 1e-12))

        out = {
            "idx": idx, "word": r["word"], "label": r["label"],
            "pred": pred, "correct": bool(correct), "gen_text": gen_text,
            "answer_slot": int(answer_slot),
            "steps": steps, "p_yes": p_yes, "p_no": p_no,
            "p_yes_2way": p_yes_2way,
        }
        with open(args.out_file, "a") as f:
            f.write(json.dumps(out) + "\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()