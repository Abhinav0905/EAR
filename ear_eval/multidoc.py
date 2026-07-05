"""Multi-document / enterprise-scale experiment.

Builds ONE combined index over several corpora and answers each corpus's own questions
against it WITHOUT a document filter, so retrieval must discriminate the answer document
from a large distractor pool. Because qids match the single-document runs, single-doc vs
multi-doc degradation is a paired per-question comparison.

  python -m ear_eval.multidoc --docs wmp,arxiv,misalignment --judge-runs 3
"""
import argparse
import json
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config, corpora, judge as judgemod, llm, pipelines, slm as slmmod, stats
from .run import CORPORA, CORE_VARIANTS, LockedSlm, variant_fn, _metrics

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

N_BY_CORPUS = {"arxiv": 119, "misalignment": 100, "wmp": 100}


def load_corpus_questions(c, chunks):
    cfg = CORPORA[c]
    n = N_BY_CORPUS.get(c, 100)
    if cfg["source"] == "wmp_golden":
        cur = corpora.load_wmp_golden(cfg["golden"], c)
        extra = corpora.generate_questions(c, chunks, cfg["title"], max(0, n - len(cur)), cache=True)
        qs = (cur + extra)[:n]
    else:
        qs = corpora.generate_questions(c, chunks, cfg["title"], n, cache=True)[:n]
    for q in qs:
        q["src"] = c
    return qs


def build_combined_store(docs, rebuild=False):
    persist = config.WORK_DIR / "chroma" / ("multidoc_" + "_".join(docs))
    col = "multidoc"
    emb = llm.TitanEmbeddings()
    all_chunks, questions = [], []
    for c in docs:
        cfg = CORPORA[c]
        _, chunks = corpora.build_store(c, cfg["pdf"], trim_refs=cfg["trim_refs"])  # cached per-corpus
        for ch in chunks:
            all_chunks.append({**ch, "src": c})
        questions += load_corpus_questions(c, chunks)
    if not rebuild and persist.exists():
        vs = Chroma(persist_directory=str(persist), collection_name=col, embedding_function=emb)
        try:
            if vs._collection.count() >= len(all_chunks):
                print(f"[multidoc] reusing combined store ({vs._collection.count()} vectors)")
                return vs, all_chunks, questions
        except Exception:
            pass
    if persist.exists():
        shutil.rmtree(persist)
    print(f"[multidoc] embedding {len(all_chunks)} chunks from {len(docs)} docs into one index...")
    docs_ = [Document(page_content=c["text"],
                      metadata={"src": c["src"], "page": c["page"], "doc_id": c["src"]}) for c in all_chunks]
    vs = Chroma.from_documents(docs_, embedding=emb, persist_directory=str(persist), collection_name=col)
    print(f"[multidoc] stored {len(docs_)} vectors")
    return vs, all_chunks, questions


def run(docs, judge_runs=3, variants=None, max_workers=5):
    variants = variants or CORE_VARIANTS
    print(f"\n{'='*70}\nMULTI-DOC: {docs} | one combined index, no doc filter\n{'='*70}")
    vs, all_chunks, questions = build_combined_store(docs)
    from collections import Counter
    print(f"[multidoc] {len(all_chunks)} chunks, {len(questions)} questions "
          f"(by src: {dict(Counter(q['src'] for q in questions))})")

    slms = {}
    if "ear_full" in variants:
        pool = [all_chunks[i] for i in np.linspace(0, len(all_chunks) - 1, num=min(config.TRAIN_CHUNKS, len(all_chunks))).astype(int)]
        events, ecost = slmmod.gen_training_events(pool, "combined enterprise corpus")
        s = slmmod.EARSlm(name="multidoc"); s.create_adapter("multidoc_main"); s.train("multidoc_main", events)
        slms["main"] = {"slm": LockedSlm(s), "aid": "multidoc_main"}
        print(f"[multidoc] trained EAR adapter on pooled chunks (events ${ecost:.3f})")

    ckpt = config.RESULTS_DIR / "multidoc_checkpoint.jsonl"
    done = set()
    if ckpt.exists():
        for l in ckpt.read_text().splitlines():
            try: done.add(json.loads(l)["key"])
            except Exception: pass
    wlock = threading.Lock(); fh = ckpt.open("a")

    def task(variant, q):
        key = f"{variant}|{q['id']}"
        if key in done:
            return
        try:
            fn = variant_fn(variant, vs, slms)
            res = fn(q)
            item = {"question": q["question"], "category": q["category"], "reference_answer": q["reference_answer"],
                    "expected_pages": q.get("expected_pages", []), "expected_source": q.get("src", ""),
                    "answer": res["answer"], "retrieved": res["retrieved"]}
            jr = judgemod.judge(item, model=config.JUDGE_MODEL, runs=judge_runs)
        except Exception as e:
            print(f"  [ERR] {key}: {type(e).__name__}: {str(e)[:120]}"); return
        # was the top retrieved chunk from the correct source document?
        top_src = res["retrieved"][0].get("src") if res.get("retrieved") else None
        rec = {"key": key, "corpus": "multidoc", "variant": variant, "qid": q["id"], "src": q.get("src"),
               "category": q["category"], "answerable": q.get("answerable", True),
               "metrics": _metrics(res), "judge_cost": jr["judge_cost"],
               "retrieval_avg": jr["retrieval_avg"], "generation_avg": jr["generation_avg"],
               "overall": jr["overall"], "top_src_correct": (top_src == q.get("src")),
               "answer": res["answer"][:800]}
        with wlock:
            fh.write(json.dumps(rec) + "\n"); fh.flush()

    for v in variants:
        todo = [q for q in questions if f"{v}|{q['id']}" not in done]
        print(f"  [{v}] running {len(todo)}...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(lambda q: task(v, q), todo))
        print(f"  [{v}] done in {time.time()-t0:.0f}s")
    fh.close()
    summarize(docs)


def summarize(docs):
    ckpt = config.RESULTS_DIR / "multidoc_checkpoint.jsonl"
    rows = [json.loads(l) for l in ckpt.read_text().splitlines()]
    byv = defaultdict(list)
    for r in rows:
        byv[r["variant"]].append(r)
    print("\n=== MULTI-DOC RESULT (combined index, no filter) ===")
    print(f"{'variant':16s} {'Gen':>5s} {'Ret':>5s} {'calls':>5s} {'top-src✓':>8s}")
    ear = {r["qid"]: r["generation_avg"] for r in byv.get("ear_full", [])}
    for v in CORE_VARIANTS:
        rs = byv.get(v, [])
        if not rs:
            continue
        g = np.mean([r["generation_avg"] for r in rs]); rt = np.mean([r["retrieval_avg"] for r in rs])
        lc = np.mean([r["metrics"]["llm_calls"] for r in rs]); ts = np.mean([1 if r.get("top_src_correct") else 0 for r in rs])
        print(f"{v:16s} {g:>5.2f} {rt:>5.2f} {lc:>5.1f} {ts*100:>7.0f}%")
    # EAR vs baselines significance on multidoc
    print("\n  EAR vs baseline (gen Δ, Wilcoxon):")
    for b in [x for x in CORE_VARIANTS if x != "ear_full"]:
        bq = {r["qid"]: r["generation_avg"] for r in byv.get(b, [])}
        common = sorted(set(ear) & set(bq))
        if not common:
            continue
        a = [ear[q] for q in common]; c = [bq[q] for q in common]
        w = stats.paired_wilcoxon(a, c)
        print(f"    vs {b:16s} Δ={w['mean_diff']:+.3f} p={w['p']:.3f}")
    print(f"[out] {ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="wmp,arxiv,misalignment")
    ap.add_argument("--judge-runs", type=int, default=3)
    ap.add_argument("--variants", default="core")
    args = ap.parse_args()
    variants = CORE_VARIANTS if args.variants == "core" else [v.strip() for v in args.variants.split(",")]
    run([d.strip() for d in args.docs.split(",")], judge_runs=args.judge_runs, variants=variants)


if __name__ == "__main__":
    main()
