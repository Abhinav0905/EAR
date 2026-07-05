"""Orchestrator: per-corpus build -> questions -> EAR train -> pipelines -> judge(3x) -> aggregate+significance.

Checkpointed (resumable) and threaded. Run:
  cd /Users/mac001/Documents/Patent/Patent_poc
  python -m ear_eval.run --corpus arxiv --n 100 --variants core
  python -m ear_eval.run --corpus arxiv --n 4 --variants core --smoke   # cheap end-to-end check
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config, corpora, judge as judgemod, pipelines, slm as slmmod, stats

WMP = "/Users/mac001/Documents/WMP-CRIS"
CORPORA = {
    "arxiv": {"pdf": f"{WMP}/2606.24824v1.pdf",
              "title": "Bi-CFM: Inverse Problems of Chaotic Systems (arXiv:2606.24824v1)",
              "trim_refs": True, "source": "auto"},
    "misalignment": {"pdf": f"{WMP}/2606.24251v1.pdf",
                     "title": "Probing the Misaligned Thinking Process of Language Models (arXiv:2606.24251v1)",
                     "trim_refs": True, "source": "auto"},
    "wmp": {"pdf": f"{WMP}/pge-wmp.pdf", "title": "PG&E 2026-2028 Wildfire Mitigation Plan",
            "trim_refs": False, "source": "wmp_golden",
            "golden": f"{WMP}/evaluation/trial1_baseline/golden_test_set.json"},
}

CORE_VARIANTS = ["simple", "reranked_simple", "agentic", "self_rag", "flare", "ear_full"]
ABLATION_VARIANTS = ["ear_rank16", "ear_noadapter", "ear_norewrite", "ear_norerank",
                     "ear_nocoverage", "ear_tau03", "ear_tau05", "ear_tau09"]


class LockedSlm:
    """Thread-safe proxy so concurrent pipeline threads share one torch SLM safely."""
    def __init__(self, slm):
        self._slm = slm
        self._lock = threading.Lock()

    def generate(self, *a, **k):
        with self._lock:
            return self._slm.generate(*a, **k)

    def generate_batch(self, *a, **k):
        with self._lock:
            return self._slm.generate_batch(*a, **k)


def build_slms(corpus, chunks, title, needed):
    """Create/train the EAR SLM instances required by the requested variants."""
    out = {}
    needs_main = any(v in needed for v in
                     ["ear_full", "ear_norewrite", "ear_norerank", "ear_nocoverage",
                      "ear_tau03", "ear_tau05", "ear_tau09"])
    events = None
    if needs_main or "ear_rank16" in needed:
        events_path = config.WORK_DIR / f"{corpus}_events.json"
        if events_path.exists():
            events = json.loads(events_path.read_text())
            print(f"[{corpus}] reusing {len(events)} cached LoRA training events")
        else:
            events, ecost = slmmod.gen_training_events(chunks, title)
            events_path.write_text(json.dumps(events))
            print(f"[{corpus}] generated {len(events)} LoRA training events (${ecost:.3f})")
    if needs_main:
        s = slmmod.EARSlm(name="main")
        aid = f"{corpus}_main"
        s.create_adapter(aid)
        m = s.train(aid, events)
        print(f"[{corpus}] trained main adapter rank8 q,v: {m}")
        out["main"] = {"slm": LockedSlm(s), "aid": aid,
                       "base_params": s.base_param_count(), "adapter_params": s.adapter_param_count(aid)}
    if "ear_rank16" in needed:
        s = slmmod.EARSlm(rank=16, alpha=32, targets=["q", "v", "k", "o"], name="rank16")
        aid = f"{corpus}_rank16"
        s.create_adapter(aid)
        m = s.train(aid, events)
        print(f"[{corpus}] trained rank16 q,v,k,o adapter: {m}")
        out["rank16"] = {"slm": LockedSlm(s), "aid": aid,
                         "base_params": s.base_param_count(), "adapter_params": s.adapter_param_count(aid)}
    if "ear_noadapter" in needed:
        s = slmmod.EARSlm(no_adapter=True, name="noadapter")
        out["noadapter"] = {"slm": LockedSlm(s), "aid": None,
                            "base_params": s.base_param_count(), "adapter_params": 0}
    return out


def variant_fn(name, vs, slms, gen_model=None):
    """Return a callable q->result for a variant name. gen_model routes the pipeline's
    generator calls (None => pipelines fall back to config.GEN_MODEL)."""
    if name == "simple":
        return lambda q: pipelines.simple_rag(vs, q["question"], gen_model=gen_model)
    if name == "reranked_simple":
        return lambda q: pipelines.reranked_simple_rag(vs, q["question"], gen_model=gen_model)
    if name == "agentic":
        return lambda q: pipelines.agentic_rag(vs, q["question"], gen_model=gen_model)
    if name == "self_rag":
        return lambda q: pipelines.self_rag(vs, q["question"], gen_model=gen_model)
    if name == "flare":
        return lambda q: pipelines.flare(vs, q["question"], gen_model=gen_model)
    if name == "ear_full":
        m = slms["main"]
        return lambda q: pipelines.ear_lora(vs, q["question"], m["slm"], m["aid"], gen_model=gen_model)
    if name == "ear_rank16":
        m = slms["rank16"]
        return lambda q: pipelines.ear_lora(vs, q["question"], m["slm"], m["aid"], gen_model=gen_model)
    if name == "ear_noadapter":
        m = slms["noadapter"]
        return lambda q: pipelines.ear_lora(vs, q["question"], m["slm"], m["aid"], gen_model=gen_model)
    if name == "ear_norewrite":
        m = slms["main"]
        return lambda q: pipelines.ear_lora(vs, q["question"], m["slm"], m["aid"], gen_model=gen_model, use_rewrite=False)
    if name == "ear_norerank":
        m = slms["main"]
        return lambda q: pipelines.ear_lora(vs, q["question"], m["slm"], m["aid"], gen_model=gen_model, use_rerank=False)
    if name == "ear_nocoverage":
        m = slms["main"]
        return lambda q: pipelines.ear_lora(vs, q["question"], m["slm"], m["aid"], gen_model=gen_model, use_coverage=False)
    if name.startswith("ear_tau"):
        tau = {"ear_tau03": 0.3, "ear_tau05": 0.5, "ear_tau09": 0.9}[name]
        m = slms["main"]
        return lambda q: pipelines.ear_lora(vs, q["question"], m["slm"], m["aid"], gen_model=gen_model, cov_threshold=tau)
    raise ValueError(name)


def _metrics(res):
    return {
        "llm_calls": len(res["calls"]),
        "slm_calls": res["extra"].get("slm_calls", 0),
        "gen_cost_usd": sum(c["cost"] for c in res["calls"]),
        "tokens": sum(c["input_tokens"] + c["output_tokens"] for c in res["calls"]),
        "wall_ms": res["wall_ms"],
        "iterations": res["extra"].get("iterations"),
    }


def run_corpus(corpus, n, variants, judge_runs=None, gen_model=None, smoke=False, max_workers=5, tag=None):
    cfg = CORPORA[corpus]
    judge_runs = judge_runs or config.JUDGE_RUNS
    gen_model = gen_model or config.GEN_MODEL
    tag = tag or (corpus + ("_haiku" if gen_model == config.HAIKU else ""))
    print(f"\n{'='*70}\nCORPUS {corpus} | n={n} | variants={variants} | gen={gen_model.split('.')[-1][:18]}\n{'='*70}")

    vs, chunks = corpora.build_store(corpus, cfg["pdf"], trim_refs=cfg["trim_refs"])

    if cfg["source"] == "wmp_golden":
        curated = corpora.load_wmp_golden(cfg["golden"], corpus)
        topup = max(0, n - len(curated))
        extra = corpora.generate_questions(corpus, chunks, cfg["title"], topup, cache=True) if topup else []
        questions = curated + extra
    else:
        questions = corpora.generate_questions(corpus, chunks, cfg["title"], n, cache=True)
    # keep all (may exceed n with supplemental negative controls)
    print(f"[{corpus}] {len(questions)} questions")

    slms = build_slms(corpus, chunks, cfg["title"], set(variants))

    ckpt = config.RESULTS_DIR / f"{tag}_checkpoint.jsonl"
    done = set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    wlock = threading.Lock()
    fh = ckpt.open("a")

    def task(variant, q):
        key = f"{variant}|{q['id']}"
        if key in done:
            return None
        try:
            fn = variant_fn(variant, vs, slms, gen_model=gen_model)
            res = fn(q)
            item = {"question": q["question"], "category": q["category"],
                    "reference_answer": q["reference_answer"], "expected_pages": q.get("expected_pages", []),
                    "expected_source": q.get("expected_source", ""), "answer": res["answer"],
                    "retrieved": res["retrieved"]}
            jr = judgemod.judge(item, model=config.JUDGE_MODEL, runs=judge_runs)
        except Exception as e:
            print(f"  [ERR] {key}: {type(e).__name__}: {str(e)[:140]} (will retry on resume)", flush=True)
            return None
        rec = {"key": key, "corpus": corpus, "variant": variant, "gen_model": gen_model,
               "qid": q["id"], "question": q["question"], "category": q["category"],
               "answerable": q.get("answerable", True),
               "metrics": _metrics(res), "judge_cost": jr["judge_cost"],
               "retrieval_avg": jr["retrieval_avg"], "retrieval_avg_std": jr["retrieval_avg_std"],
               "generation_avg": jr["generation_avg"], "generation_avg_std": jr["generation_avg_std"],
               "overall": jr["overall"], "rubric_means": jr["rubric_means"],
               "answer": res["answer"][:1500],
               "reference_answer": q.get("reference_answer", ""),
               "retrieved_trim": [{"page": c.get("page"), "text": (c.get("text") or "")[:300]}
                                  for c in res["retrieved"][:6]]}
        with wlock:
            fh.write(json.dumps(rec) + "\n"); fh.flush()
        return rec

    for variant in variants:
        todo = [q for q in questions if f"{variant}|{q['id']}" not in done]
        if not todo:
            print(f"  [{variant}] already complete ({len(questions)} cached)")
            continue
        print(f"  [{variant}] running {len(todo)} questions...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=(1 if smoke else max_workers)) as ex:
            list(ex.map(lambda q: task(variant, q), todo))
        print(f"  [{variant}] done in {time.time()-t0:.0f}s")
    fh.close()
    return tag, slms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPORA) + ["all"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--variants", default="core", help="core | all | ablation | comma-list")
    ap.add_argument("--judge-runs", type=int, default=config.JUDGE_RUNS)
    ap.add_argument("--gen", default="sonnet", choices=["sonnet", "haiku"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.variants == "core":
        variants = CORE_VARIANTS
    elif args.variants == "ablation":
        variants = ["ear_full"] + ABLATION_VARIANTS
    elif args.variants == "all":
        variants = CORE_VARIANTS + ABLATION_VARIANTS
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    gen_model = config.HAIKU if args.gen == "haiku" else config.SONNET
    corp = list(CORPORA) if args.corpus == "all" else [args.corpus]
    for c in corp:
        run_corpus(c, args.n, variants, judge_runs=args.judge_runs, gen_model=gen_model, smoke=args.smoke)
    print("\nDone. Aggregate with: python -m ear_eval.report")


if __name__ == "__main__":
    main()
