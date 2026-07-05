"""Multi-doc retriever-precision diagnostic (no generation, no judge).

Answers the reviewer question: on the combined index, is any quality gap a
DOCUMENT-DISCRIMINATION problem (retriever pulls chunks from the wrong document)
or a WITHIN-DOCUMENT one (right document, wrong chunk)? We measure, at the
retriever level only, how often the correct source document is retrieved:

  * top-1 correct : is retrieved[0] from the question's source document?
  * recall@5      : does any of the top-5 come from the correct document?
  * precision@5   : what fraction of the top-5 come from the correct document?

for three retriever families that underlie the six pipelines:
  * dense     (Simple / Agentic / Self-RAG / FLARE first hop): Titan top-k
  * reranked  (Reranked-Simple): Titan fetch -> cross-encoder rerank
  * ear       (EAR): LoRA query-rewrite -> Titan fetch -> SLM rerank

Cost is ~embeddings only (+ one small event-gen if the EAR adapter is trained).

  python -m ear_eval.multidoc_retrieval --docs wmp,arxiv,misalignment
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np

from . import config, pipelines, slm as slmmod
from .multidoc import build_combined_store

TOP = 5


def _dense(vs, q):
    return pipelines.retrieve(vs, q["question"], config.TOP_K)[:TOP]


def _reranked(vs, q):
    cands = pipelines.retrieve(vs, q["question"], config.RERANK_FETCH)
    return pipelines.ce_rerank(q["question"], cands, TOP)


def _ear(vs, q, slm, aid):
    seen, alld = set(), []

    def add(docs):
        for d in docs:
            k = d["text"][:120]
            if k not in seen:
                seen.add(k); alld.append(d)

    queries = [q["question"]]
    rq = slm.generate(aid, f"rewrite query: {q['question']}", max_len=64)
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
    return sorted(alld, key=lambda c: c["slm_score"], reverse=True)[:TOP]


def _metrics(retrieved, src):
    srcs = [c.get("src") for c in retrieved]
    top1 = 1.0 if (srcs and srcs[0] == src) else 0.0
    recall5 = 1.0 if any(s == src for s in srcs) else 0.0
    prec5 = (sum(1 for s in srcs if s == src) / len(srcs)) if srcs else 0.0
    return top1, recall5, prec5


def run(docs, train_ear=True):
    print(f"\n{'='*70}\nMULTI-DOC RETRIEVER PRECISION: {docs}\n{'='*70}")
    vs, all_chunks, questions = build_combined_store(docs)
    from collections import Counter
    print(f"[retr] {len(all_chunks)} chunks, {len(questions)} questions "
          f"(by src: {dict(Counter(q['src'] for q in questions))})")

    retrievers = {"dense": lambda q: _dense(vs, q),
                  "reranked": lambda q: _reranked(vs, q)}

    if train_ear:
        events_path = config.WORK_DIR / "multidoc_events.json"
        if events_path.exists():
            events = json.loads(events_path.read_text())
            print(f"[retr] reusing {len(events)} cached multidoc LoRA events")
        else:
            pool = [all_chunks[i] for i in np.linspace(
                0, len(all_chunks) - 1, num=min(config.TRAIN_CHUNKS, len(all_chunks))).astype(int)]
            events, ecost = slmmod.gen_training_events(pool, "combined enterprise corpus")
            events_path.write_text(json.dumps(events))
            print(f"[retr] generated {len(events)} multidoc LoRA events (${ecost:.3f})")
        s = slmmod.EARSlm(name="multidoc_retr")
        s.create_adapter("multidoc_main")
        m = s.train("multidoc_main", events)
        print(f"[retr] trained EAR adapter: {m}")
        retrievers["ear"] = lambda q: _ear(vs, q, s, "multidoc_main")

    # accumulate: agg[retriever][src] -> list of (top1, recall5, prec5)
    agg = {r: defaultdict(list) for r in retrievers}
    t0 = time.time()
    for i, q in enumerate(questions):
        for rname, rfn in retrievers.items():
            try:
                retrieved = rfn(q)
            except Exception as e:
                print(f"  [ERR] {rname} q{i}: {type(e).__name__}: {str(e)[:80]}")
                continue
            mt = _metrics(retrieved, q["src"])
            agg[rname][q["src"]].append(mt)
            agg[rname]["_all"].append(mt)
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(questions)} ({time.time()-t0:.0f}s)")

    # summarize
    out = {"docs": docs, "top_n": TOP, "n_questions": len(questions),
           "by_src_count": dict(Counter(q["src"] for q in questions)), "retrievers": {}}
    print(f"\n{'='*70}\nRESULT (correct-source-document retrieval on combined index)\n{'='*70}")
    for rname in retrievers:
        out["retrievers"][rname] = {}
        print(f"\n[{rname}]")
        print(f"  {'source':14s} {'n':>4s} {'top-1✓':>8s} {'recall@5':>9s} {'prec@5':>8s}")
        srcs = [s for s in agg[rname] if s != "_all"] + ["_all"]
        for src in srcs:
            rows = agg[rname][src]
            if not rows:
                continue
            arr = np.array(rows)
            t1, r5, p5 = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
            out["retrievers"][rname][src] = {"n": len(rows), "top1": round(float(t1), 4),
                                             "recall5": round(float(r5), 4), "prec5": round(float(p5), 4)}
            label = "OVERALL" if src == "_all" else src
            print(f"  {label:14s} {len(rows):>4d} {t1*100:>7.1f}% {r5*100:>8.1f}% {p5*100:>7.1f}%")

    outpath = config.RESULTS_DIR / "multidoc_retrieval.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\n[out] {outpath}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="wmp,arxiv,misalignment")
    ap.add_argument("--no-ear", action="store_true")
    args = ap.parse_args()
    run([d.strip() for d in args.docs.split(",")], train_ear=not args.no_ear)


if __name__ == "__main__":
    main()
