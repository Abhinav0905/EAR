"""Adapter-value / domain-shift experiment (reviewer item ⑤).

Tests whether the ingestion-trained LoRA adapter carries corpus-specific value beyond the
frozen base model, by running EAR on a TARGET corpus with three controller configurations:
  matched     — adapter trained on the target corpus's own ingestion events
  mismatched  — adapter trained on a DIFFERENT (source) corpus's events  (domain shift)
  noadapter   — frozen base flan-t5, no adapter
If matched > mismatched ≈ noadapter, the adapter specializes to its corpus; if all three tie,
the adapter is a design affordance rather than a quality lever (consistent with §VII).

  python -m ear_eval.adapter_value --target wmp --source arxiv --n 100
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config, corpora, judge as judgemod, pipelines, slm as slmmod
from .run import CORPORA, LockedSlm, _metrics


def _load_questions(corpus, chunks, n):
    cfg = CORPORA[corpus]
    if cfg["source"] == "wmp_golden":
        curated = corpora.load_wmp_golden(cfg["golden"], corpus)
        extra = corpora.generate_questions(corpus, chunks, cfg["title"], max(0, n - len(curated)), cache=True)
        return (curated + extra)[:n]
    return corpora.generate_questions(corpus, chunks, cfg["title"], n, cache=True)[:n]


def run(target="wmp", source="arxiv", n=100, judge_runs=3, max_workers=5):
    tcfg, scfg = CORPORA[target], CORPORA[source]
    print(f"\n{'='*70}\nADAPTER-VALUE: target={target}  source(mismatch)={source}  n={n}\n{'='*70}")

    vs_t, chunks_t = corpora.build_store(target, tcfg["pdf"], trim_refs=tcfg["trim_refs"])
    _, chunks_s = corpora.build_store(source, scfg["pdf"], trim_refs=scfg["trim_refs"])
    questions = _load_questions(target, chunks_t, n)
    print(f"[{target}] {len(questions)} questions")

    # matched adapter (target events)
    ev_t, _ = slmmod.gen_training_events(chunks_t, tcfg["title"])
    s_m = slmmod.EARSlm(name="matched"); s_m.create_adapter("m"); print("  matched:", s_m.train("m", ev_t))
    # mismatched adapter (source events)
    ev_s, _ = slmmod.gen_training_events(chunks_s, scfg["title"])
    s_x = slmmod.EARSlm(name="mismatched"); s_x.create_adapter("x"); print("  mismatched:", s_x.train("x", ev_s))
    # no adapter (frozen base)
    s_b = slmmod.EARSlm(no_adapter=True, name="noadapter")

    configs = [("matched", LockedSlm(s_m), "m"), ("mismatched", LockedSlm(s_x), "x"),
               ("noadapter", LockedSlm(s_b), None)]

    ckpt = config.RESULTS_DIR / f"{target}_adaptertest_checkpoint.jsonl"
    done = set()
    if ckpt.exists():
        for l in ckpt.read_text().splitlines():
            try: done.add(json.loads(l)["key"])
            except Exception: pass
    wlock = threading.Lock(); fh = ckpt.open("a")

    def task(name, slm, aid, q):
        key = f"{name}|{q['id']}"
        if key in done: return
        try:
            res = pipelines.ear_lora(vs_t, q["question"], slm, aid)
            item = {"question": q["question"], "category": q["category"],
                    "reference_answer": q["reference_answer"], "expected_pages": q.get("expected_pages", []),
                    "expected_source": q.get("expected_source", ""), "answer": res["answer"], "retrieved": res["retrieved"]}
            jr = judgemod.judge(item, model=config.JUDGE_MODEL, runs=judge_runs)
        except Exception as e:
            print(f"  [ERR] {key}: {type(e).__name__}: {str(e)[:120]}"); return
        rec = {"key": key, "corpus": f"{target}_adaptertest", "variant": name, "qid": q["id"],
               "category": q["category"], "answerable": q.get("answerable", True),
               "metrics": _metrics(res), "judge_cost": jr["judge_cost"],
               "retrieval_avg": jr["retrieval_avg"], "generation_avg": jr["generation_avg"],
               "overall": jr["overall"], "answer": res["answer"][:800]}
        with wlock:
            fh.write(json.dumps(rec) + "\n"); fh.flush()

    for name, slm, aid in configs:
        todo = [q for q in questions if f"{name}|{q['id']}" not in done]
        print(f"  [{name}] running {len(todo)}...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(lambda q: task(name, slm, aid, q), todo))
        print(f"  [{name}] done in {time.time()-t0:.0f}s")
    fh.close()

    # quick summary + significance
    import numpy as np
    from . import stats
    rows = [json.loads(l) for l in ckpt.read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["variant"], {})[r["qid"]] = r["generation_avg"]
    print("\n=== ADAPTER-VALUE RESULT (generation axis) ===")
    for v in ["matched", "mismatched", "noadapter"]:
        if v in by:
            vals = list(by[v].values())
            print(f"  {v:11s} Gen={np.mean(vals):.3f}±{np.std(vals,ddof=1):.2f} (n={len(vals)})")
    if "matched" in by and "mismatched" in by:
        common = sorted(set(by["matched"]) & set(by["mismatched"]))
        a = [by["matched"][q] for q in common]; b = [by["mismatched"][q] for q in common]
        w = stats.paired_wilcoxon(a, b)
        print(f"  matched − mismatched: Δ={w['mean_diff']:+.3f}  Wilcoxon p={w['p']:.4f}")
    if "matched" in by and "noadapter" in by:
        common = sorted(set(by["matched"]) & set(by["noadapter"]))
        a = [by["matched"][q] for q in common]; b = [by["noadapter"][q] for q in common]
        w = stats.paired_wilcoxon(a, b)
        print(f"  matched − noadapter:  Δ={w['mean_diff']:+.3f}  Wilcoxon p={w['p']:.4f}")
    print(f"[out] {ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="wmp", choices=list(CORPORA))
    ap.add_argument("--source", default="arxiv", choices=list(CORPORA))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--judge-runs", type=int, default=3)
    args = ap.parse_args()
    run(target=args.target, source=args.source, n=args.n, judge_runs=args.judge_runs)


if __name__ == "__main__":
    main()
