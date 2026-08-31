#!/usr/bin/env python3
# scripts/fix_prosqa_step43.py
"""
One-off fix: prosqa_contrastive_dlig.py was accidentally run with 44 instead
of 43 in --target_steps for the main run; a corrective rerun with the
correct 43 was done separately, sharded (3 unmerged shards). This script:

  1. Merges the 3 corrective shards (outputs/prosqa/prosqa_dlig_missing43_shard{0,1,2}.jsonl).
  2. For each example (matched by idx) in the main prosqa_dlig.jsonl:
       - drops the steps_data entry with step==44 (the mistaken run),
       - inserts the corrective steps_data entry with step==43 in its place
         (from the merged shards, matched by idx),
       - re-sorts steps_data by step.
  3. Writes the patched file to a NEW path (does not overwrite the input) so
     you can diff/sanity-check before replacing the original.

Usage:
  python -m scripts.fix_prosqa_step43 \
      --main_file outputs/prosqa/prosqa_dlig.jsonl \
      --shard_glob "outputs/prosqa/prosqa_dlig_missing43_shard*.jsonl" \
      --out_file outputs/prosqa/prosqa_dlig_fixed.jsonl

Then, once you've spot-checked the output:
  mv outputs/prosqa/prosqa_dlig_fixed.jsonl outputs/prosqa/prosqa_dlig.jsonl
"""
import json
import glob
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_file", default="outputs/prosqa/prosqa_dlig.jsonl")
    ap.add_argument("--shard_glob",
                    default="outputs/prosqa/prosqa_dlig_missing43_shard*.jsonl")
    ap.add_argument("--out_file", default="outputs/prosqa/prosqa_dlig_fixed.jsonl")
    ap.add_argument("--bad_step", type=int, default=44)
    ap.add_argument("--good_step", type=int, default=43)
    args = ap.parse_args()

    shard_files = sorted(glob.glob(args.shard_glob))
    if not shard_files:
        raise SystemExit(f"[ERROR] no shard files matched {args.shard_glob!r}")
    print(f"[INFO] merging {len(shard_files)} shard(s): {shard_files}")

    # patch[idx] = the step==good_step steps_data entry for that example
    patch = {}
    n_shard_rows = 0
    n_missing_good_step = 0
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_shard_rows += 1
                idx = rec["idx"]
                hit = next((sd for sd in rec["steps_data"]
                           if sd.get("step") == args.good_step), None)
                if hit is None:
                    n_missing_good_step += 1
                    continue
                if idx in patch:
                    print(f"[WARN] idx={idx} appears in multiple corrective shards; "
                          f"keeping the first occurrence.")
                    continue
                patch[idx] = hit
    print(f"[INFO] {n_shard_rows} corrective rows merged, "
          f"{len(patch)} usable step={args.good_step} patches "
          f"({n_missing_good_step} rows had no step={args.good_step} entry)")

    n_main = n_had_bad = n_patched = n_no_patch_available = 0
    with open(args.main_file) as fin, open(args.out_file, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_main += 1
            idx = rec["idx"]

            before = len(rec["steps_data"])
            rec["steps_data"] = [sd for sd in rec["steps_data"]
                                 if sd.get("step") != args.bad_step]
            had_bad = len(rec["steps_data"]) < before
            if had_bad:
                n_had_bad += 1

            if idx in patch:
                rec["steps_data"].append(patch[idx])
                n_patched += 1
            elif had_bad:
                n_no_patch_available += 1
                print(f"[WARN] idx={idx} had step={args.bad_step} removed but "
                      f"no step={args.good_step} patch was found -- that "
                      f"example is now MISSING a step in the output.")

            rec["steps_data"].sort(key=lambda sd: sd.get("step", 0))
            fout.write(json.dumps(rec) + "\n")

    print(f"\n[RESULT] {n_main} examples processed")
    print(f"  had step={args.bad_step} removed : {n_had_bad}")
    print(f"  patched with step={args.good_step} : {n_patched}")
    if n_no_patch_available:
        print(f"  MISSING patch (see warnings above) : {n_no_patch_available}")
    print(f"  -> {args.out_file}")
    print(f"\nSanity-check the output, then:\n"
          f"  mv {args.out_file} {args.main_file}")


if __name__ == "__main__":
    main()
