#!/usr/bin/env python3
# inspect_wic_train.py
"""
Locate where the yes-bias enters. Checks:
  1. Label balance in wic_train.jsonl (should be ~50/50).
  2. The tokenized answer span: does "### Yes." vs "### No." differ only in the
     one answer token, or does tokenization make them different lengths (which
     would give the diffusion loss an uneven target and bias the easy class)?
  3. Whether <|endoftext|> maps to the single eos id (50256) or is being written
     as literal text (which would mean the model never actually sees an EOS to
     learn to stop).

Usage:
  python inspect_train.py --train_jsonl data/wic_train.jsonl
"""

import json
import argparse
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/wic_train.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.train_jsonl) if l.strip()]
    n = len(rows)

    # 1. label balance, inferred from the output text (### Yes / ### No)
    ans = Counter()
    for r in rows:
        o = r["output"]
        if "Yes" in o:
            ans["Yes"] += 1
        elif "No" in o:
            ans["No"] += 1
        else:
            ans["?"] += 1
    print(f"n train            : {n}")
    print(f"answer balance     : {dict(ans)}")
    print(f"  Yes-rate         : {ans['Yes']/n:.3f}  (want ~0.500)")
    print("-" * 44)

    # 2 + 3. tokenization of the two answer targets
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"[WARN] tokenizer unavailable ({e}); skipping token inspection.")
        return

    eos_id = tok.eos_token_id
    print(f"gpt2 eos id        : {eos_id}")
    for out in ['### Yes.<|endoftext|>', '### No.<|endoftext|>']:
        ids = tok(out, add_special_tokens=False)["input_ids"]
        toks = [tok.decode([t]) for t in ids]
        has_eos = eos_id in ids
        print(f"\n  target {out!r}")
        print(f"    ids   : {ids}")
        print(f"    toks  : {toks}")
        print(f"    len   : {len(ids)}   eos present as id: {has_eos}")
        if not has_eos:
            print(f"    [!] <|endoftext|> did NOT tokenize to the eos id — the model")
            print(f"        never sees a real EOS, so it can't learn to stop.")

    # length parity check
    y = tok('### Yes.<|endoftext|>', add_special_tokens=False)["input_ids"]
    nn = tok('### No.<|endoftext|>', add_special_tokens=False)["input_ids"]
    print("-" * 44)
    if len(y) != len(nn):
        print(f"[!] Yes target len ({len(y)}) != No target len ({len(nn)}).")
        print("    Unequal answer-span lengths give the diffusion loss an uneven")
        print("    target across classes and can drive a bias. Consider padding")
        print("    the shorter to match, or a single-token answer scheme.")
    else:
        print(f"[OK] Yes/No targets same length ({len(y)} tokens); differ only in")
        print("     the answer token — no length-induced class bias.")


if __name__ == "__main__":
    main()