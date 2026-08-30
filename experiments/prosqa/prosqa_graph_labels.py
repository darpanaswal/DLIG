#!/usr/bin/env python3
# experiments/prosqa_graph_labels.py
"""
prosqa_graph_labels.py — per-example structural labels for ProsQA DLIG analysis.

INPUT : bucketed jsonl from bucket_prosqa.py
        (idx, question, gold, gen, pred_answer, bucket, q_subject, options, ...)
OUTPUT: jsonl, one record per example:
        idx, group, subject, gold_option, wrong_option, k_gold (gold path hops),
        spans: list of labeled char spans in the question string:
            {id, kind: "edge"|"root"|"question",
             src, dst, span: [start, end),
             on_gold_path, hop,          # hop: 1-based along root->gold path
             on_wrong_path,              # on shortest path root->wrong option (if any)
             in_model_chain}             # cited in model's own generated chain
        model_chain: normalized concept sequence extracted from `gen`
        parse_ok, notes

Why: ProsQA questions carry an explicit ontology DAG ("Every X is a Y." edges +
one root fact "<Subj> is a <Z>."). The gold answer is reachable from the root
concept by a unique-ish chain; everything else is distractor. This gives
token-level ground truth for attribution: DLIG mass SHOULD sit on gold-path
edges for successes, and (faithfulness hypothesis) on the model's OWN cited
chain for wrong_valid failures.

Concept matching is edit-tolerant because the model emits malformed concepts
("jelp" for "jelpus", "zumpusus" for "zumpus"): two concepts match iff their
lowercase-alpha forms are equal, or one is a prefix of the other with the
shorter side >= 4 chars.
"""

import json
import argparse
from collections import deque, Counter

import re

# ---------- parsing ----------

# ontology edge: "Every X is a Y."   (allow "a" / "an")
_EDGE = re.compile(r"Every\s+(\w+)\s+is\s+an?\s+(\w+)\s*\.")
# root fact: "<CapName> is a <concept>."  (named entities are capitalized)
_ROOT = re.compile(r"\b([A-Z][a-z]+)\s+is\s+an?\s+(\w+)\s*\.")
# question tail: "Is <Subj> a <A> or <B>?"
_Q = re.compile(r"Is\s+(\w+)\s+an?\s+(\w+)\s+or\s+(\w+)\s*\?")


def norm(c: str) -> str:
    return re.sub(r"[^a-z]", "", c.lower())


def concept_match(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 4 and long_.startswith(short)


def parse_question(q: str):
    """Return (edges, roots, qspan_info). Spans are (start, end) char offsets in q."""
    edges = [{"src": m.group(1).lower(), "dst": m.group(2).lower(),
              "span": [m.start(), m.end()]} for m in _EDGE.finditer(q)]
    edge_spans = [tuple(e["span"]) for e in edges]

    roots = []
    for m in _ROOT.finditer(q):
        # skip matches that live inside an edge span ("Every ..." never matches
        # _ROOT since "Every" is filtered by concept use below, but be safe)
        if any(s <= m.start() < e for (s, e) in edge_spans):
            continue
        name = m.group(1)
        if name.lower() == "every" or name.lower() == "is":
            continue
        roots.append({"name": name.lower(), "concept": m.group(2).lower(),
                      "span": [m.start(), m.end()]})

    qm = _Q.search(q)
    qinfo = None
    if qm:
        qinfo = {"subject": qm.group(1).lower(),
                 "options": [qm.group(2).lower(), qm.group(3).lower()],
                 "span": [qm.start(), qm.end()]}
    return edges, roots, qinfo


# ---------- graph ----------

def bfs_path(edges, starts, goal):
    """
    Shortest node path from ANY of `starts` -> goal over directed edges.
    Multi-source: ProsQA subjects can carry several root facts ("Alex is a
    wumpus. Alex is a lorpus.") and the gold chain may depart from any of them;
    single-source BFS from the first root fact wrongly reports unreachable.
    Returns (edge_index_path, start_concept_used) or (None, None).
    """
    adj = {}
    for i, e in enumerate(edges):
        adj.setdefault(e["src"], []).append((e["dst"], i))
    prev = {}                             # node -> (prev_node, via_edge_idx)
    dq = deque()
    for s in starts:
        if s not in prev:
            prev[s] = (None, None)
            dq.append(s)
    while dq:
        u = dq.popleft()
        if u == goal:
            path = []
            while prev[u][0] is not None:
                pu, ei = prev[u]
                path.append(ei)
                u = pu
            return path[::-1], u          # u is now the start that reached goal
        for v, ei in adj.get(u, []):
            if v not in prev:
                prev[v] = (u, ei)
                dq.append(v)
    return None, None


# ---------- model chain ----------

def extract_model_chain(gen: str, subject: str):
    """
    Concept sequence the model itself asserted in `gen`:
      "<Subj> is a X."  -> chain starts at X
      "Every X is a Y." -> appends Y when X matches current tail (tolerant),
                           else starts a fresh segment (chain may derail).
    Returns flat list of normalized concepts in citation order.
    """
    chain = []
    for m in re.finditer(r"(?:Every\s+(\w+)|\b(\w+))\s+is\s+an?\s+(\w+)", gen):
        src = m.group(1) or m.group(2)
        dst = m.group(3)
        if m.group(2) and concept_match(m.group(2), subject):
            src = None  # "<Subj> is a X": no source concept, X seeds the chain
        elif m.group(2):
            continue     # some other named entity / noise
        if src is not None:
            chain.append(norm(src))
        chain.append(norm(dst))
    # dedupe consecutive repeats
    out = []
    for c in chain:
        if not out or not concept_match(out[-1], c):
            out.append(c)
    return out


def mark_model_chain(edges, root_concept, chain):
    """
    An edge (src -> dst) is "in the model chain" iff some consecutive concept
    pair in the chain matches it (tolerant). The root concept prepends the
    chain if the chain's head matches something reachable from it.
    """
    seq = list(chain)
    hits = set()
    for i, e in enumerate(edges):
        for a, b in zip(seq, seq[1:]):
            if concept_match(a, e["src"]) and concept_match(b, e["dst"]):
                hits.add(i)
                break
    return hits


# ---------- main ----------

def label_example(row):
    q = row["question"]
    edges, roots, qinfo = parse_question(q)
    notes = []

    if qinfo is None:
        return None, "question-parse-fail"
    subject = qinfo["subject"]
    options = qinfo["options"]

    # gold option = final concept of the gold answer
    gold_words = re.findall(r"[a-zA-Z]+", row["gold"])
    gold_option = gold_words[-1].lower() if gold_words else ""
    if gold_option not in options:
        return None, "gold-not-in-options"
    wrong_option = options[0] if options[1] == gold_option else options[1]

    # root facts for the queried subject (can be several; use all as BFS sources)
    subj_roots = [r for r in roots if r["name"] == subject]
    if not subj_roots:
        return None, ("root-fact-missing",
                      {"edges": edges, "roots": roots, "q": q})
    start_concepts = [r["concept"] for r in subj_roots]

    gold_path, gold_start = bfs_path(edges, start_concepts, gold_option)
    if gold_path is None:
        return None, ("gold-path-unreachable",
                      {"edges": edges, "roots": roots, "subject": subject,
                       "gold_option": gold_option, "q": q})
    wrong_path, wrong_start = bfs_path(edges, start_concepts, wrong_option)

    chain = extract_model_chain(row.get("gen", ""), subject)
    chain_hits = mark_model_chain(edges, gold_start, chain)

    hop_of = {ei: h + 1 for h, ei in enumerate(gold_path)}
    wrong_set = set(wrong_path) if wrong_path else set()

    spans = []
    for i, e in enumerate(edges):
        spans.append({
            "id": i, "kind": "edge", "src": e["src"], "dst": e["dst"],
            "span": e["span"],
            "on_gold_path": i in hop_of, "hop": hop_of.get(i, 0),
            "on_wrong_path": i in wrong_set,
            "in_model_chain": i in chain_hits,
        })
    # Emit EVERY root fact as a labeled span (subject's and other entities');
    # non-subject roots are distractor facts and belong in the FACT denominator.
    # on_gold_path marks only the subject root whose concept starts the gold
    # chain; in_model_chain marks the subject root the model's chain departs from.
    for r in roots:
        rid = len(spans)
        is_subj = r["name"] == subject
        spans.append({
            "id": rid, "kind": "root", "src": r["name"], "dst": r["concept"],
            "span": r["span"], "is_subject": is_subj,
            "on_gold_path": is_subj and r["concept"] == gold_start,
            "hop": 0,
            "on_wrong_path": is_subj and wrong_path is not None
                             and r["concept"] == wrong_start,
            "in_model_chain": is_subj and bool(chain)
                              and concept_match(chain[0], r["concept"]),
        })
    spans.append({"id": len(spans), "kind": "question", "src": subject,
                  "dst": None, "span": qinfo["span"],
                  "on_gold_path": False, "hop": 0,
                  "on_wrong_path": False, "in_model_chain": False})

    bucket = row["bucket"]
    group = ("success" if bucket in ("correct", "concept_only")
             else "fail" if bucket == "wrong_valid" else "off")

    rec = {
        "idx": row["idx"], "bucket": bucket, "group": group,
        "subject": subject, "gold_option": gold_option,
        "wrong_option": wrong_option,
        "gold": row["gold"],
        "k_gold": len(gold_path),
        "k_wrong": len(wrong_path) if wrong_path else -1,
        "n_edges": len(edges),
        "model_chain": chain,
        "spans": spans,
        "question": q,
    }
    return rec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", required=True, help="bucket_prosqa.py output jsonl")
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    n = 0
    fails = Counter()
    kdist = Counter()
    fail_path = args.out_file.replace(".jsonl", "_failures.jsonl")
    with open(args.buckets) as fin, open(args.out_file, "w") as fout, \
         open(fail_path, "w") as ffail:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rec, err = label_example(row)
            if err:
                # err is (reason, diagnostics) for parse/graph failures
                reason, diag = err if isinstance(err, tuple) else (err, {})
                fails[reason] += 1
                ffail.write(json.dumps(
                    {"idx": row.get("idx"), "reason": reason, **diag}) + "\n")
                continue
            kdist[rec["k_gold"]] += 1
            fout.write(json.dumps(rec) + "\n")
            n += 1

    print(f"[GRAPH] labeled {n} examples -> {args.out_file}")
    if fails:
        print(f"[GRAPH] skipped: {dict(fails)}  (diagnostics -> {fail_path})")
    print(f"[GRAPH] gold-path hop distribution: {dict(sorted(kdist.items()))}")


if __name__ == "__main__":
    main()