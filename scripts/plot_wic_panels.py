#!/usr/bin/env python3
# scripts/plot_wic_panels.py
"""
One-off: regenerate the 4 qualitative WiC panel figures from outputs_old
(no_throw, no_wall, yes_channel, yes_love) against the current
outputs/wic/wic_dlig.jsonl. idx is just the row position in
wic_test_raw.jsonl (stable across runs), so this looks each word's idx up
rather than requiring you to hardcode it -- EXCEPT wall, which the CLI
docstring in helpers/analyze_wic.py already documents as idx=208, so that one
is hardcoded to the known-correct value rather than re-derived.

CAUTION -- multiple WiC rows can share the same target word (a WSD dataset
tests the same word across different sentence pairs). A naive "first row with
this word" lookup is NOT reliable: an earlier run of this script found
word='throw' -> idx=192 (pred=1, correct=False) and word='wall' -> idx=145
(pred=1, correct=True) as the FIRST matches, but "no_throw"/"no_wall" both
require pred=0 (the "No"/different-sense response) -- both were wrong, and
neither matched the docstring's documented idx=208 for wall. So: filter by
word AND the required (pred, correct) group, not word alone.

VERIFIED INDICES (from an actual run against outputs/wic/wic_dlig.jsonl,
T=64 / 32 target_steps -- re-run and update this note if wic_dlig.jsonl is
regenerated from a different eval, since idx-to-content mapping only depends
on wic_test_raw.jsonl but the pred/correct fields depend on the checkpoint
and generation run):
    no_wall      -> idx=208  (hardcoded; documented in analyze_wic.py's own
                              --panel usage example, NOT re-derived here)
                              label=0 pred=0 correct=True
    no_throw     -> idx=264  (group-filtered search result, verified: label=0
                              pred=0 correct=True; first-match-by-word-alone
                              idx=192 was WRONG -- pred=1, not 0)
    yes_channel  -> idx=168  (first-match happened to already satisfy pred=1,
                              correct=True -- this is Figure 1's own example;
                              label=1 pred=1 correct=True)
    yes_love     -> idx=116  (first-match happened to already satisfy pred=1,
                              correct=True; label=1 pred=1 correct=True)

Usage:
  python -m scripts.plot_wic_panels
"""
from helpers.analyze_wic import load, plot_layer_panels

# name -> (word, required pred, hardcoded idx or None to search)
EXAMPLES = {
    "no_throw":    ("throw",   0, None),
    "no_wall":     ("wall",    0, 208),   # documented in analyze_wic.py's docstring
    "yes_channel": ("channel", 1, None),
    "yes_love":    ("love",    1, None),
}
DLIG_FILE = "outputs/wic/wic_dlig.jsonl"
OUT_DIR = "outputs/wic/figs"
PANEL_LAYERS = [0, 2, 4, 8, 12, 16, 18, 20, 22]  # matches Figure 1 (DLIG_NeurIPS.pdf) exactly,
                                                  # NOT analyze_wic.py's CLI default of
                                                  # [0,4,8,12,16,20,22] (7 layers, missing 2 & 18)


def find_row(rows, word, want_pred, hardcoded_idx):
    if hardcoded_idx is not None:
        for r in rows:
            if r["idx"] == hardcoded_idx:
                return r
        print(f"[WARN] hardcoded idx={hardcoded_idx} not found in {DLIG_FILE}")
        return None
    # require the correct predicted-answer GROUP (not just word match) --
    # WiC tests the same word across multiple sentence pairs, so word alone
    # is not a unique or reliable selector.
    candidates = [r for r in rows if r.get("word") == word
                 and r.get("pred") == want_pred and r.get("correct")]
    if not candidates:
        # fall back to pred-matching but not necessarily correct, so you at
        # least get the right response-type group instead of nothing
        candidates = [r for r in rows if r.get("word") == word
                     and r.get("pred") == want_pred]
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"[WARN] {len(candidates)} candidates for word={word!r} pred={want_pred} "
              f"-- picking idx={candidates[0]['idx']}, but this is NOT verified "
              f"against the original paper figure (only wall's idx=208 is documented).")
    return candidates[0]


def main():
    rows = load(DLIG_FILE)

    for name, (word, want_pred, hardcoded_idx) in EXAMPLES.items():
        row = find_row(rows, word, want_pred, hardcoded_idx)
        if row is None:
            print(f"[WARN] no pred={want_pred} example with word={word!r} "
                  f"found in {DLIG_FILE} -- skipping {name}")
            continue
        idx = row["idx"]
        print(f"{name}: word={word!r} idx={idx} label={row.get('label')} "
              f"pred={row.get('pred')} correct={row.get('correct')}")
        out_file = f"{OUT_DIR}/correct_{name}.png"
        # min_frac=0.08, cols=3: matches Figure 1's stated rule exactly --
        # "hides tokens with |s_t^(l)[i]| < 0.08 max_i|s_t^(l)[i]| (keeping
        # >= 3 largest)" and its 3x3 layer grid (9 layers / 3 cols).
        plot_layer_panels(row, PANEL_LAYERS, out_file, per_step=False,
                          min_frac=0.08, cols=3)


if __name__ == "__main__":
    main()
