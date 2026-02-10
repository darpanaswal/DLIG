#!/usr/bin/env python3
"""
Sample 500 harmful prompts from CatQA and 500 benign prompts from Alpaca-Cleaned,
and write a JSON file:

{
  "harmful": ["..."],  # CatQA Question strings ONLY
  "benign":  ["..."]   # Alpaca instruction (+ optional input)
}

Defaults (Hugging Face):
- CatQA:  declare-lab/CategoricalHarmfulQA
- Alpaca: yahma/alpaca-cleaned

Usage:
  python make_harmful_benign_json.py --out prompts.json --seed 42
"""

import ast
import json
import argparse
from datasets import load_dataset
from typing import Any, Dict, List, Optional


def _as_nonempty_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _first_existing_key(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            return k
    return None


# ---------------------------
# Alpaca prompt extraction
# ---------------------------
def build_alpaca_prompt(ex: Dict[str, Any]) -> str:
    instr = _as_nonempty_str(ex.get("instruction", ""))
    inp = _as_nonempty_str(ex.get("input", ""))

    if instr and inp:
        return f"{instr}\n\nInput: {inp}"
    return instr or inp


# ---------------------------
# CatQA "Question only" extraction
# ---------------------------
def extract_catqa_question(ex: Dict[str, Any]) -> str:
    """
    Robustly extract ONLY the Question string from CatQA.

    Handles cases where:
      1) Dataset has a proper 'Question' field.
      2) Dataset has 'question' (lowercase).
      3) The example stores a Python-dict-like string, e.g.
         "{'Category': ..., 'Question': '...'}"
         in a 'text' / 'prompt' / 'question' field.

    Returns "" if not found.
    """
    # Case 1/2: direct field
    for k in ["Question", "question"]:
        if k in ex and ex[k] is not None:
            return _as_nonempty_str(ex[k])

    # Case 3: dict-like string stored in some field
    candidate_key = _first_existing_key(ex, ["text", "prompt", "query", "instruction", "data"])
    if candidate_key:
        s = _as_nonempty_str(ex.get(candidate_key, ""))
        if s.startswith("{") and "Question" in s:
            try:
                obj = ast.literal_eval(s)  # safe parse for Python literals
                if isinstance(obj, dict) and "Question" in obj:
                    return _as_nonempty_str(obj.get("Question", ""))
            except Exception:
                pass

    # Final fallback: scan all string fields for dict-like payloads containing "Question"
    for v in ex.values():
        s = _as_nonempty_str(v)
        if s.startswith("{") and "Question" in s:
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, dict) and "Question" in obj:
                    return _as_nonempty_str(obj.get("Question", ""))
            except Exception:
                continue

    return ""


def filter_catqa_english(ds):
    possible_lang_keys = ["language", "lang", "locale"]
    lang_key = None
    for k in possible_lang_keys:
        if k in ds.column_names:
            lang_key = k
            break
    if lang_key is None:
        return ds

    def is_english(ex):
        v = _as_nonempty_str(ex.get(lang_key, "")).lower()
        return v == "english" or v.startswith("en")

    return ds.filter(is_english)


def sample_list(ds, n: int, seed: int, builder) -> List[str]:
    ds = ds.shuffle(seed=seed)
    out: List[str] = []
    # Iterate deterministically over shuffled rows until we collect n non-empty
    for ex in ds:
        p = _as_nonempty_str(builder(ex))
        if p:
            out.append(p)
        if len(out) >= n:
            break
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    ap.add_argument("--n", type=int, default=500, help="Samples per class")
    ap.add_argument(
        "--catqa_repo",
        type=str,
        default="declare-lab/CategoricalHarmfulQA",
        help="HF dataset repo for CatQA",
    )
    ap.add_argument(
        "--alpaca_repo",
        type=str,
        default="yahma/alpaca-cleaned",
        help="HF dataset repo for Alpaca-Cleaned",
    )
    args = ap.parse_args()

    catqa = load_dataset(args.catqa_repo)
    alpaca = load_dataset(args.alpaca_repo)

    catqa_split = "train" if "train" in catqa else list(catqa.keys())[0]
    alpaca_split = "train" if "train" in alpaca else list(alpaca.keys())[0]

    catqa_ds = filter_catqa_english(catqa[catqa_split])
    alpaca_ds = alpaca[alpaca_split]

    harmful = sample_list(catqa_ds, args.n, args.seed, extract_catqa_question)
    benign = sample_list(alpaca_ds, args.n, args.seed, build_alpaca_prompt)

    payload = {"harmful": harmful, "benign": benign}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(harmful)} harmful + {len(benign)} benign prompts -> {args.out}")


if __name__ == "__main__":
    main()
