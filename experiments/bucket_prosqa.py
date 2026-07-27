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

import os
import re
import json
import argparse
from collections import Counter


BUCKET_EXPLANATIONS = {
    "correct": (
        "Exact string match on the gold answer: predicted subject and option both "
        "match, and the generated answer sentence matches the gold wording verbatim."
    ),
    "concept_only": (
        "Right subject, right option (A/B), but not a verbatim match -- extra text "
        "or formatting (e.g. stray '###' markers) around an otherwise correct answer "
        "makes it fail strict exact-match while still being conceptually correct."
    ),
    "wrong_valid": (
        "Right subject, and the model committed to one of the two offered options, "
        "but picked the wrong one. A genuine reasoning error, not a parsing failure."
    ),
    "subj_wrong": (
        "The model's final sentence names a different (or malformed) subject than "
        "the one asked about, so whatever concept it landed on is coincidental -- "
        "gated to this bucket regardless of whether the option happens to be correct."
    ),
    "concept_invalid": (
        "Right subject, but the predicted concept is not even one of the two offered "
        "options -- the reasoning chain derailed into an unrelated category, so there "
        "is no valid option to score as right or wrong."
    ),
}
BUCKET_ORDER = ("correct", "concept_only", "wrong_valid", "concept_invalid", "subj_wrong")


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


def summary_path_for(out_file: str) -> str:
    root, _ext = os.path.splitext(out_file)
    return f"{root}_summary.txt"


def write_summary(summary_file, in_file, out_file, n, counts, examples,
                   n_unparsed, n_subjfail):
    success = counts["correct"] + counts["concept_only"]
    fail    = counts["wrong_valid"]
    off     = counts["subj_wrong"] + counts["concept_invalid"]

    lines = []
    lines.append("ProsQA bucketed task-level performance")
    lines.append("=" * 60)
    lines.append(f"input : {in_file}")
    lines.append(f"labels: {out_file}")
    lines.append(f"n={n}  (option-parse fail: {n_unparsed}; subj mismatch: {n_subjfail})")
    lines.append("")
    lines.append("Counts by bucket")
    lines.append("-" * 60)
    for b in BUCKET_ORDER:
        pct = 100 * counts[b] / n if n else 0.0
        lines.append(f"  {b:16}: {counts[b]:4}  ({pct:5.1f}%)")
    lines.append("")
    lines.append("Groups (as used for DLIG contrastive filtering)")
    lines.append("-" * 60)
    lines.append(f"  success (correct U concept_only): {success:4}  ({100*success/n:.1f}%)" if n else "  success: 0")
    lines.append(f"  fail    (wrong_valid)           : {fail:4}  ({100*fail/n:.1f}%)" if n else "  fail: 0")
    lines.append(f"  off     (subj_wrong+concept_inv): {off:4}  ({100*off/n:.1f}%)" if n else "  off: 0")
    lines.append("")
    lines.append("Bucket definitions & examples")
    lines.append("=" * 60)
    for b in BUCKET_ORDER:
        lines.append(f"\n[{b}]")
        lines.append(BUCKET_EXPLANATIONS[b])
        ex = examples.get(b)
        if ex is not None:
            q_tail = ex["question"][-120:]
            lines.append(f"  example (idx={ex['idx']}):")
            lines.append(f"    question (tail): ...{q_tail}")
            lines.append(f"    gold            : {ex['gold']}")
            lines.append(f"    gen             : {ex['gen']}")
            lines.append(f"    pred_answer     : {ex['pred_answer']}")
            lines.append(f"    q_subject={ex['q_subject']}  pred_subject={ex['pred_subject']}  "
                          f"subj_match={ex['subj_match']}")
            lines.append(f"    options={ex['options']}  predicted_option={ex['predicted_option']}  "
                          f"answer_valid={ex['answer_valid']}")
            lines.append(f"    exact_match={ex['exact_match']}  concept_match={ex['concept_match']}")
        else:
            lines.append("  (no example in this run)")

    with open(summary_file, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file",  required=True)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    counts = Counter()
    examples = {}
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
            examples.setdefault(lab["bucket"], lab)
            n += 1
            fout.write(json.dumps(lab) + "\n")

    success = counts["correct"] + counts["concept_only"]
    fail    = counts["wrong_valid"]
    off     = counts["subj_wrong"] + counts["concept_invalid"]

    print(f"[BUCKET] n={n}  (option-parse fail: {n_unparsed}; subj mismatch: {n_subjfail})")
    for b in BUCKET_ORDER:
        print(f"  {b:16}: {counts[b]:4}  ({100*counts[b]/n:5.1f}%)")
    print(f"[GROUP] success(correct U concept_only): {success} ({100*success/n:.1f}%)")
    print(f"[GROUP] fail   (wrong_valid)           : {fail} ({100*fail/n:.1f}%)")
    print(f"[GROUP] off    (subj_wrong+concept_inv): {off} ({100*off/n:.1f}%)")
    print(f"  labels -> {args.out_file}")

    summary_file = summary_path_for(args.out_file)
    write_summary(summary_file, args.in_file, args.out_file, n, counts, examples,
                   n_unparsed, n_subjfail)
    print(f"  summary -> {summary_file}")


if __name__ == "__main__":
    main()