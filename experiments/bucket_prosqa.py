#!/usr/bin/env python3
# bucket_prosqa.py  (v2: subject-match gate)
"""
Per-example labels for ProsQA DiffuGPT-M eval, for contrastive DLIG filtering.

INPUT : eval jsonl (question, gold, gen, pred_answer, exact, concept_match)
OUTPUT: labelled jsonl + fields: idx, q_subject, pred_subject, subj_match,
        options, predicted_option, answer_valid, bucket

Bucket logic (subject gate FIRST):
    NOT subj_match   -> off_manifold   (model misread which entity is asked)
    exact_match      -> correct
    concept_match    -> concept_only
    answer_valid     -> wrong_valid
    else             -> off_manifold

Rationale: ProsQA fixes the queried subject ("Is <subj> a A or B?"). If the
model's answer is about the wrong subject it misunderstood the problem, so the
right concept is coincidental -> not a success. Gate on subject before concept.

Contrast for DLIG:
    success = correct U concept_only
    fail    = wrong_valid
    DROP    = off_manifold
"""

import re
import json
import argparse
from collections import Counter


def final_concept(s):
    # last alphabetic token, lowercased (matches eval_prosqa.final_concept)
    w = re.findall(r"[a-zA-Z]+", s)
    return w[-1].lower() if w else ""


# "Is <subj> a <optA> or <optB>?"
_Q = re.compile(r"Is (\w+) a (\w+) or (\w+)\s*\?")


def q_parse(question):
    m = _Q.search(question)
    if not m:
        return None, None
    subj = m.group(1).lower()
    opts = (m.group(2).lower(), m.group(3).lower())
    return subj, opts


def pred_subject(pred):
    # ProsQA subjects are capitalized names (Sally, Bob, Max...).
    # Take first such name anywhere in pred (tolerates "### Bob is a numpus").
    # If none -> "" -> guaranteed subj mismatch -> off_manifold (degenerate gen).
    m = re.search(r"\b([A-Z][a-z]+)\b", pred)
    return m.group(1).lower() if m else ""


def label_row(row, idx):
    q     = row["question"]
    pred  = row["pred_answer"]
    exact = bool(row["exact"])
    conc  = bool(row["concept_match"])

    q_subj, opts = q_parse(q)
    pc = final_concept(pred)
    ps = pred_subject(pred)

    subj_match = (q_subj is not None) and (ps == q_subj)
    answer_valid = (opts is not None) and (pc in opts)
    predicted_option = pc if answer_valid else None

    # subject gate first: wrong entity = distinct failure mode
    if not subj_match:
        bucket = "subj_wrong"          # misread which entity is queried
    elif exact:
        bucket = "correct"
    elif conc:
        bucket = "concept_only"
    elif answer_valid:
        bucket = "wrong_valid"         # subj ok, picked wrong valid option
    else:
        bucket = "concept_invalid"     # subj ok, concept not in {A,B} (degenerate/invented)

    return {
        "idx": idx,
        "question": q,
        "gold": row["gold"],
        "gen": row.get("gen", ""),
        "pred_answer": pred,
        "exact_match": exact,
        "concept_match": conc,
        "q_subject": q_subj,
        "pred_subject": ps,
        "subj_match": subj_match,
        "options": list(opts) if opts else None,
        "answer_valid": answer_valid,
        "predicted_option": predicted_option,
        "bucket": bucket,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file",  required=True)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    counts = Counter()
    n = n_unparsed = n_subjfail = 0
    with open(args.in_file) as fin, open(args.out_file, "w") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lab = label_row(row, i)
            if lab["options"] is None:
                n_unparsed += 1
            if not lab["subj_match"]:
                n_subjfail += 1
            counts[lab["bucket"]] += 1
            n += 1
            fout.write(json.dumps(lab) + "\n")

    success = counts["correct"] + counts["concept_only"]
    fail    = counts["wrong_valid"]
    off     = counts["subj_wrong"] + counts["concept_invalid"]

    print(f"[BUCKET] n={n}  (option-parse fail: {n_unparsed}; subj mismatch: {n_subjfail})")
    for b in ("correct", "concept_only", "wrong_valid", "concept_invalid", "subj_wrong"):
        print(f"  {b:16}: {counts[b]:4}  ({100*counts[b]/n:5.1f}%)")
    print(f"[GROUP] success(correct U concept_only): {success} ({100*success/n:.1f}%)")
    print(f"[GROUP] fail   (wrong_valid)           : {fail} ({100*fail/n:.1f}%)")
    print(f"[GROUP] off    (subj_wrong+concept_inv): {off} ({100*off/n:.1f}%)")
    print(f"  labels -> {args.out_file}")


if __name__ == "__main__":
    main()