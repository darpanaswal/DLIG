#!/usr/bin/env python3
# experiments/prosqa/prepare_graph_labels.py
"""
Chains bucket_prosqa.py -> prosqa_graph_labels.py into one call, with fixed
default paths matching the established pattern already used by
prosqa_contrastive_dlig.py / args/prosqa_dlig.sh:

    data/prosqa_eval_full.jsonl
        -> [bucket_prosqa.label_row]      -> outputs/prosqa/prosqa_buckets_full.jsonl
        -> [prosqa_graph_labels.label_example] -> outputs/prosqa/prosqa_graph_labels.jsonl

No required args for the normal case -- just:
    python -m experiments.prosqa.prepare_graph_labels

All three intermediate/output paths are still overridable for one-off runs
(e.g. a different eval-mode input, per bucket_prosqa.py's --in_file /
data/prosqa_eval_{mode}.jsonl convention).
"""
import argparse

from experiments.prosqa.bucket_prosqa import (
    label_row, BUCKET_ORDER, summary_path_for, write_summary,
)
from experiments.prosqa.prosqa_graph_labels import label_example
from collections import Counter
import json


def run_bucket_prosqa(in_file, out_file):
    counts = Counter()
    examples = {}
    n = n_unparsed = n_subjfail = 0
    with open(in_file) as fin, open(out_file, "w") as fout:
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
    fail = counts["wrong_valid"]
    off = counts["subj_wrong"] + counts["concept_invalid"]

    print(f"[BUCKET] n={n}  (option-parse fail: {n_unparsed}; subj mismatch: {n_subjfail})")
    for b in BUCKET_ORDER:
        print(f"  {b:16}: {counts[b]:4}  ({100*counts[b]/n:5.1f}%)")
    print(f"[GROUP] success(correct U concept_only): {success} ({100*success/n:.1f}%)")
    print(f"[GROUP] fail   (wrong_valid)           : {fail} ({100*fail/n:.1f}%)")
    print(f"[GROUP] off    (subj_wrong+concept_inv): {off} ({100*off/n:.1f}%)")
    print(f"  labels -> {out_file}")

    summary_file = summary_path_for(out_file)
    write_summary(summary_file, in_file, out_file, n, counts, examples,
                  n_unparsed, n_subjfail)
    print(f"  summary -> {summary_file}")


def run_graph_labels(buckets_file, out_file):
    n = 0
    fails = Counter()
    kdist = Counter()
    fail_path = out_file.replace(".jsonl", "_failures.jsonl")
    with open(buckets_file) as fin, open(out_file, "w") as fout, \
         open(fail_path, "w") as ffail:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rec, err = label_example(row)
            if err:
                reason, diag = err if isinstance(err, tuple) else (err, {})
                fails[reason] += 1
                ffail.write(json.dumps(
                    {"idx": row.get("idx"), "reason": reason, **diag}) + "\n")
                continue
            kdist[rec["k_gold"]] += 1
            fout.write(json.dumps(rec) + "\n")
            n += 1

    print(f"[GRAPH] labeled {n} examples -> {out_file}")
    if fails:
        print(f"[GRAPH] skipped: {dict(fails)}  (diagnostics -> {fail_path})")
    print(f"[GRAPH] gold-path hop distribution: {dict(sorted(kdist.items()))}")


def main():
    ap = argparse.ArgumentParser(
        description="bucket_prosqa.py + prosqa_graph_labels.py, chained, "
                    "with fixed default paths."
    )
    ap.add_argument("--in_file", default="data/prosqa_eval_full.jsonl",
                    help="Raw eval jsonl (question/gold/gen/pred_answer/exact/concept_match).")
    ap.add_argument("--buckets_file", default="outputs/prosqa/prosqa_buckets_full.jsonl",
                    help="Intermediate bucket_prosqa.py output / prosqa_graph_labels.py input.")
    ap.add_argument("--out_file", default="outputs/prosqa/prosqa_graph_labels.jsonl",
                    help="Final graph-labels output, consumed by "
                         "prosqa_contrastive_dlig.py's --graph_labels.")
    args = ap.parse_args()

    print(f"[STEP 1/2] bucket: {args.in_file} -> {args.buckets_file}")
    run_bucket_prosqa(args.in_file, args.buckets_file)

    print(f"\n[STEP 2/2] graph labels: {args.buckets_file} -> {args.out_file}")
    run_graph_labels(args.buckets_file, args.out_file)

    print(f"\nDONE. Pass --graph_labels {args.out_file} to prosqa_contrastive_dlig.py.")


if __name__ == "__main__":
    main()
