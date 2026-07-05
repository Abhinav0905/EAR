"""STEP 4 diagnostic: why are EAR's document-level retrieval rates identical to dense
in multidoc_retrieval.json? Replicates the EXACT _dense / _ear paths from
multidoc_retrieval.py and, per question, compares top-5 chunk IDs and inspects the
LoRA rewrite. Report-only (no manuscript edits, no relabeling). Embeddings cost only.

  python -m ear_eval.multidoc_retrieval_diag
"""
import hashlib
import json

import numpy as np

from . import config, pipelines, slm as slmmod
from .multidoc import build_combined_store

TOP = 5


def _key(c):
    return hashlib.md5((c.get("text") or "").encode("utf-8")).hexdigest()[:12]


def _dense_top5(vs, q):
    return [_key(c) for c in pipelines.retrieve(vs, q["question"], config.TOP_K)[:TOP]]


def _ear_top5_and_rewrite(vs, q, slm, aid):
    """Exact replica of multidoc_retrieval._ear, but returns (top5 keys, rewrite string)."""
    seen, alld = set(), []

    def add(docs):
        for d in docs:
            k = d["text"][:120]
            if k not in seen:
                seen.add(k); alld.append(d)

    rq = slm.generate(aid, f"rewrite query: {q['question']}", max_len=64)
    queries = [q["question"]]
    if rq and len(rq.strip()) > 3:
        queries.append(rq.strip())
    for query in queries:
        add(pipelines.retrieve(vs, query, config.RERANK_FETCH))
    outs = slm.generate_batch(
        aid, [f"rank relevance: query: {q['question']} document: {c['text'][:200]}" for c in alld], max_len=8)

    def score(o):
        o = (o or "").lower()
        return 1.0 if ("relevant" in o and "irrelevant" not in o) else (0.0 if "irrelevant" in o else 0.5)
    for c, o in zip(alld, outs):
        c["slm_score"] = score(o)
    ranked = sorted(alld, key=lambda c: c["slm_score"], reverse=True)[:TOP]
    return [_key(c) for c in ranked], (rq or ""), [c.get("slm_score") for c in ranked]


def run(docs=("wmp", "arxiv", "misalignment")):
    docs = list(docs)
    print(f"\n{'='*70}\nSTEP 4 DIAG: EAR vs dense retrieval on combined index\n{'='*70}")
    vs, all_chunks, questions = build_combined_store(docs)
    print(f"[diag] {len(all_chunks)} chunks, {len(questions)} questions")

    events_path = config.WORK_DIR / "multidoc_events.json"
    events = json.loads(events_path.read_text())
    print(f"[diag] reusing {len(events)} cached multidoc LoRA events")
    s = slmmod.EARSlm(name="multidoc_diag")
    s.create_adapter("multidoc_main")
    s.train("multidoc_main", events)

    order_sensitive_diff = 0
    order_insensitive_diff = 0
    rewrite_differs = 0
    rewrite_used = 0            # rewrite passed the len>3 gate (actually appended as a query)
    slm_all_neutral = 0        # top-5 all scored 0.5 (degenerate rerank)
    examples = []
    for i, q in enumerate(questions):
        dense = _dense_top5(vs, q)
        ear, rewrite, scores = _ear_top5_and_rewrite(vs, q, s, "multidoc_main")
        if ear != dense:
            order_sensitive_diff += 1
        if set(ear) != set(dense):
            order_insensitive_diff += 1
        rw = (rewrite or "").strip()
        if rw and rw.lower() != q["question"].strip().lower():
            rewrite_differs += 1
        if rw and len(rw) > 3:
            rewrite_used += 1
        if scores and all(sc == 0.5 for sc in scores):
            slm_all_neutral += 1
        if len(examples) < 5 and (ear != dense or (rw and rw.lower() != q["question"].strip().lower())):
            examples.append({"q": q["question"][:80], "rewrite": rw[:80],
                             "dense": dense, "ear": ear, "ear_scores": scores})
        if (i + 1) % 80 == 0:
            print(f"  ...{i+1}/{len(questions)}")

    n = len(questions)
    print(f"\n--- RESULTS (n={n}) ---")
    print(f"(a) top-5 chunk-ID lists differ  — order-SENSITIVE : {order_sensitive_diff}/{n}")
    print(f"    top-5 chunk-ID lists differ  — order-INSENSITIVE: {order_insensitive_diff}/{n}")
    print(f"(b) LoRA rewrites differing from original query      : {rewrite_differs}/{n}")
    print(f"    rewrites long enough to be USED (len>3)          : {rewrite_used}/{n}")
    print(f"    top-5 with all-neutral SLM rerank scores (0.5)   : {slm_all_neutral}/{n}")
    if examples:
        print("\n  examples where EAR≠dense or rewrite changed:")
        for e in examples:
            print(f"    Q: {e['q']}\n      rewrite: {e['rewrite']!r}\n      dense={e['dense']}\n      ear  ={e['ear']} scores={e['ear_scores']}")

    out = {"n": n, "top5_differ_order_sensitive": order_sensitive_diff,
           "top5_differ_order_insensitive": order_insensitive_diff,
           "rewrites_differ_from_original": rewrite_differs, "rewrites_used_len_gt3": rewrite_used,
           "top5_all_neutral_slm_scores": slm_all_neutral}
    outpath = config.RESULTS_DIR / "multidoc_retrieval_diag.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\n[out] {outpath}")
    return out


if __name__ == "__main__":
    run()
