"""QASPER (scientific-paper QA) public benchmark — Tier 2.

Builds a per-paper Titan/Chroma store, runs the same pipelines + harmonized judge,
and also reports the native QASPER Answer-F1 so numbers are comparable to published
Self-RAG / FLARE results.

  cd /Users/mac001/Documents/Patent/Patent_poc
  /path/python -m pip install datasets
  python -m ear_eval.qasper --n 100 --variants core
"""
import argparse
import collections
import json
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config, judge as judgemod, llm, pipelines, slm as slmmod
from .run import LockedSlm, variant_fn, _metrics

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# ── QASPER Answer-F1 (SQuAD-style) ────────────────────────────────────
def _norm(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def best_f1(pred, golds):
    golds = [g for g in golds if g]
    return max((token_f1(pred, g) for g in golds), default=None)


def _gold_answers(ans_field):
    """Extract gold answer strings + answerable flag from a QASPER 'answers' entry."""
    golds, answerable = [], False
    for a in ans_field.get("answer", []):
        if a.get("unanswerable"):
            golds.append("unanswerable")
            continue
        answerable = True
        if a.get("yes_no") is not None:
            golds.append("Yes" if a["yes_no"] else "No")
        if a.get("free_form_answer"):
            golds.append(a["free_form_answer"])
        if a.get("extractive_spans"):
            golds.append(" ".join(a["extractive_spans"]))
    return [g for g in golds if g] or ["unanswerable"], answerable


def load_questions(n):
    from datasets import load_dataset
    # HF dropped dataset scripts; use the auto-converted parquet revision.
    ds = load_dataset("allenai/qasper", revision="refs/convert/parquet", split="validation")
    papers, total = [], 0
    for row in ds:
        ft = row.get("full_text", {}) or {}
        sections = ft.get("section_name", []) or []
        paras = ft.get("paragraphs", []) or []
        texts = [(row.get("title", ""), row.get("abstract", ""))]
        chunks = []
        ab = row.get("abstract", "")
        if ab:
            chunks.append({"text": ab, "page": 0, "idx": 0})
        for si, plist in enumerate(paras):
            sec = sections[si] if si < len(sections) else f"sec{si}"
            for para in plist:
                if para and len(para.strip()) > 40:
                    chunks.append({"text": f"{sec}: {para}", "page": si + 1, "idx": len(chunks)})
        qas = row.get("qas", {}) or {}
        qlist = qas.get("question", []) or []
        qids = qas.get("question_id", []) or []
        answers = qas.get("answers", []) or []
        qrecs = []
        for i, q in enumerate(qlist):
            af = answers[i] if i < len(answers) else {"answer": []}
            golds, answerable = _gold_answers(af)
            qrecs.append({"id": qids[i] if i < len(qids) else f"{row['id']}_{i}",
                          "question": q, "golds": golds, "answerable": answerable,
                          "reference_answer": golds[0], "category": "qasper"})
        if not chunks or not qrecs:
            continue
        papers.append({"id": row["id"], "chunks": chunks, "qas": qrecs})
        total += len(qrecs)
        if total >= n:
            break
    return papers, total


def build_paper_store(pid, chunks):
    emb = llm.TitanEmbeddings()
    persist = config.WORK_DIR / "chroma" / f"qasper_{pid}"
    col = f"qasper_{pid}"[:60]
    if persist.exists():
        vs = Chroma(persist_directory=str(persist), collection_name=col, embedding_function=emb)
        try:
            if vs._collection.count() >= len(chunks):
                return vs
        except Exception:
            pass
    docs = [Document(page_content=c["text"], metadata={"page": c["page"], "idx": c["idx"]}) for c in chunks]
    return Chroma.from_documents(docs, embedding=emb, persist_directory=str(persist), collection_name=col)


def run(n=100, variants=None, judge_runs=1, gen_model=None, max_workers=5):
    variants = variants or ["simple", "reranked_simple", "agentic", "self_rag", "flare", "ear_full"]
    gen_model = gen_model or config.GEN_MODEL
    papers, total = load_questions(n)
    print(f"[qasper] {len(papers)} papers, {total} questions")

    stores = {p["id"]: build_paper_store(p["id"], p["chunks"]) for p in papers}

    slms = {}
    if "ear_full" in variants:
        pool = [c for p in papers for c in p["chunks"]]
        pool = [pool[i] for i in np.linspace(0, len(pool) - 1, num=min(config.TRAIN_CHUNKS, len(pool))).astype(int)]
        events, ecost = slmmod.gen_training_events(pool, "scientific papers (QASPER)")
        s = slmmod.EARSlm(name="qasper")
        s.create_adapter("qasper_main")
        s.train("qasper_main", events)
        slms["main"] = {"slm": LockedSlm(s), "aid": "qasper_main",
                        "base_params": s.base_param_count(), "adapter_params": s.adapter_param_count("qasper_main")}
        print(f"[qasper] trained EAR adapter on pooled chunks (events ${ecost:.3f})")

    ckpt = config.RESULTS_DIR / "qasper_checkpoint.jsonl"
    done = set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    wlock = threading.Lock()
    fh = ckpt.open("a")

    tasks = [(p, q, v) for p in papers for q in p["qas"] for v in variants
             if f"{v}|{q['id']}" not in done]
    print(f"[qasper] {len(tasks)} tasks to run")

    def do(t):
        p, q, v = t
        vs = stores[p["id"]]
        fn = variant_fn(v, vs, slms)
        res = fn(q)
        item = {"question": q["question"], "category": "qasper", "reference_answer": q["reference_answer"],
                "expected_pages": [], "expected_source": "", "answer": res["answer"], "retrieved": res["retrieved"]}
        jr = judgemod.judge(item, model=config.JUDGE_MODEL, runs=judge_runs)
        f1 = best_f1(res["answer"], q["golds"])
        rec = {"key": f"{v}|{q['id']}", "corpus": "qasper", "variant": v, "gen_model": gen_model,
               "qid": q["id"], "question": q["question"], "category": "qasper", "answerable": q["answerable"],
               "paper_id": p["id"], "metrics": _metrics(res), "judge_cost": jr["judge_cost"],
               "retrieval_avg": jr["retrieval_avg"], "generation_avg": jr["generation_avg"],
               "overall": jr["overall"], "rubric_means": jr["rubric_means"], "answer_f1": f1,
               "reference_answer": q["reference_answer"], "answer": res["answer"][:1500],
               "retrieved_trim": [{"page": c.get("page"), "text": (c.get("text") or "")[:300]} for c in res["retrieved"][:6]]}
        with wlock:
            fh.write(json.dumps(rec) + "\n"); fh.flush()
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(do, tasks))
    fh.close()
    print(f"[qasper] done in {time.time()-t0:.0f}s -> {ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--variants", default="core")
    ap.add_argument("--judge-runs", type=int, default=1)
    ap.add_argument("--gen", default="sonnet", choices=["sonnet", "haiku"])
    args = ap.parse_args()
    from .run import CORE_VARIANTS
    variants = CORE_VARIANTS if args.variants == "core" else [v.strip() for v in args.variants.split(",")]
    run(n=args.n, variants=variants, judge_runs=args.judge_runs,
        gen_model=(config.HAIKU if args.gen == "haiku" else config.SONNET))


if __name__ == "__main__":
    main()
