"""Cross-family judge validation: re-score the ALREADY-GENERATED answers with an
OpenAI model (gpt-4o-mini), a DIFFERENT model family from the Claude generator/judge.

This addresses the study's main construct-validity threat (§X): judge, generator,
question-generator, and adapter-teacher are all Claude-family. A cross-family judge
that (a) reproduces the between-system rankings and (b) tracks the human annotators
directly answers that objection. No answers are regenerated — we re-judge the stored
records with the identical nine-rubric protocol.

  python -m ear_eval.judge_openai --corpora wmp,arxiv,misalignment,qasper --runs 1
  python -m ear_eval.judge_openai --human      # gpt-4o-mini vs human consensus on the 49-item sheet
  python -m ear_eval.judge_openai --analyze    # summarize agreement + ranking preservation
"""
import argparse
import csv
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config, stats
from . import llm as llmmod
from .judge import JUDGE_SYSTEM, RETRIEVAL_RUBRICS, GENERATION_RUBRICS, ALL_RUBRICS, _clamp

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_PRICING = {"input": 0.15, "output": 0.60}  # USD / 1M tokens
CORE = ["simple", "reranked_simple", "agentic", "self_rag", "flare", "ear_full"]
CROSS_CKPT = config.RESULTS_DIR / "crossjudge_gpt4omini.jsonl"
HUMAN_CKPT = config.RESULTS_DIR / "crossjudge_human_gpt4omini.jsonl"

_client = None


def client():
    global _client
    if _client is None:
        import httpx
        from openai import OpenAI
        config.load_env()
        key = os.environ.get("OPENAI_KEYS") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("no OPENAI_KEYS / OPENAI_API_KEY in env")
        _client = OpenAI(api_key=key, http_client=httpx.Client(timeout=90))
    return _client


def _one_pass(item):
    chunks = item.get("retrieved", []) or []
    ctx = "\n\n---\n\n".join(
        f"[chunk {i+1}] (page {c.get('page')}) {(c.get('text') or '')[:600]}"
        for i, c in enumerate(chunks[:8])
    ) or "(no sources available; score the generation rubrics on the answer vs the reference)"
    prompt = (
        f"Question: {item['question']}\n"
        f"Category: {item.get('category','')}\n"
        f"Reference answer (ideal): {item.get('reference_answer','')}\n"
        f"Expected sources/pages: {item.get('expected_pages','')} {item.get('expected_source','')}\n\n"
        f"Retrieved context:\n{ctx}\n\n"
        f"System answer:\n{item.get('answer','')}\n\n"
        f"Score all 9 rubrics now. Output ONLY JSON."
    )
    for attempt in range(8):
        try:
            r = client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=700,
                response_format={"type": "json_object"},
            )
            text = r.choices[0].message.content
            u = r.usage
            cost = u.prompt_tokens / 1e6 * OPENAI_PRICING["input"] + \
                u.completion_tokens / 1e6 * OPENAI_PRICING["output"]
            return llmmod.extract_json(text) or {}, cost
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower() or "timeout" in msg.lower() or "connection" in msg.lower():
                time.sleep(min(30, 2 ** attempt + 0.5 * attempt)); continue
            if attempt < 2:
                time.sleep(2); continue
            raise
    raise RuntimeError("gpt-4o-mini judge failed after retries")


def judge_openai(item, runs=1):
    per = {k: [] for k in ALL_RUBRICS}
    ret_avgs, gen_avgs, cost = [], [], 0.0
    for _ in range(runs):
        sc, c = _one_pass(item); cost += c
        rs = sc.get("retrieval_scores", {}) or {}
        gs = sc.get("generation_scores", {}) or {}
        rvals = [_clamp(rs.get(k, 3)) for k in RETRIEVAL_RUBRICS]
        gvals = [_clamp(gs.get(k, 3)) for k in GENERATION_RUBRICS]
        for k, v in zip(RETRIEVAL_RUBRICS, rvals): per[k].append(v)
        for k, v in zip(GENERATION_RUBRICS, gvals): per[k].append(v)
        ret_avgs.append(float(np.mean(rvals))); gen_avgs.append(float(np.mean(gvals)))
    return {"retrieval_avg": float(np.mean(ret_avgs)), "generation_avg": float(np.mean(gen_avgs)),
            "overall": float((np.mean(ret_avgs) + np.mean(gen_avgs)) / 2),
            "rubric_means": {k: float(np.mean(v)) for k, v in per.items()},
            "judge_cost": cost}


def _item_from_record(r):
    return {"question": r["question"], "category": r.get("category", ""),
            "reference_answer": r.get("reference_answer", ""), "answer": r.get("answer", ""),
            "retrieved": [{"page": c.get("page"), "text": c.get("text")} for c in r.get("retrieved_trim", [])],
            "expected_pages": "", "expected_source": ""}


def rejudge_corpora(corpora, runs=1, max_workers=6):
    done = set()
    if CROSS_CKPT.exists():
        for l in CROSS_CKPT.read_text().splitlines():
            try: done.add(json.loads(l)["key"])
            except Exception: pass
    wlock = threading.Lock(); fh = CROSS_CKPT.open("a")

    def task(r):
        key = f"{r['corpus']}|{r['variant']}|{r['qid']}"
        if key in done:
            return
        try:
            jr = judge_openai(_item_from_record(r), runs=runs)
        except Exception as e:
            print(f"  [ERR] {key}: {type(e).__name__}: {str(e)[:100]}"); return
        rec = {"key": key, "corpus": r["corpus"], "variant": r["variant"], "qid": r["qid"],
               "gen_gpt": jr["generation_avg"], "ret_gpt": jr["retrieval_avg"], "overall_gpt": jr["overall"],
               "gen_haiku": r.get("generation_avg"), "ret_haiku": r.get("retrieval_avg"),
               "overall_haiku": r.get("overall"), "judge_cost": jr["judge_cost"]}
        with wlock:
            fh.write(json.dumps(rec) + "\n"); fh.flush()

    for corpus in corpora:
        src = config.RESULTS_DIR / f"{corpus}_checkpoint.jsonl"
        recs = [json.loads(l) for l in src.read_text().splitlines()]
        recs = [r for r in recs if r["variant"] in CORE]
        todo = [r for r in recs if f"{corpus}|{r['variant']}|{r['qid']}" not in done]
        print(f"[{corpus}] {len(recs)} core records, {len(todo)} to judge")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(task, todo))
        print(f"[{corpus}] done in {time.time()-t0:.0f}s")
    fh.close()


def rejudge_human(runs=3, max_workers=6):
    """Re-judge the 49-item human sheet with gpt-4o-mini, recovering retrieved context
    from the checkpoints so it is the SAME instrument as the corpus re-judge."""
    a1 = {row["sample_id"]: row for row in csv.DictReader(open(config.RESULTS_DIR / "human_review.csv"))}
    # annotator-2 scores live in the .xlsx (the .csv score column is empty)
    import openpyxl
    wb = openpyxl.load_workbook(config.RESULTS_DIR / "human_review_annotator2.xlsx", data_only=True)
    ws = wb.active
    xr = list(ws.iter_rows(values_only=True))
    xh = list(xr[0])
    oi, si = xh.index("orig_id"), xh.index("your_score_1to5")
    a2 = {}
    for r in xr[1:]:
        if r[oi] is not None and r[si] not in (None, ""):
            a2[str(r[oi]).strip()] = {"your_score_1to5": r[si]}
    # index checkpoints by (corpus, question, answer[:80]) to recover context
    idx = {}
    for corpus in ["wmp", "arxiv", "misalignment", "qasper"]:
        src = config.RESULTS_DIR / f"{corpus}_checkpoint.jsonl"
        if not src.exists():
            continue
        for l in src.read_text().splitlines():
            r = json.loads(l)
            if r["variant"] not in CORE:
                continue
            idx[(r["corpus"], r["question"].strip(), (r.get("answer", "") or "")[:80])] = r
            idx.setdefault((r["corpus"], r["question"].strip()), r)

    done = set()
    if HUMAN_CKPT.exists():
        for l in HUMAN_CKPT.read_text().splitlines():
            try: done.add(json.loads(l)["sample_id"])
            except Exception: pass
    wlock = threading.Lock(); fh = HUMAN_CKPT.open("a")

    def task(sid):
        if sid in done:
            return
        row = a1[sid]
        if not str(row.get("your_score_1to5") or "").strip() or sid not in a2 or a2[sid].get("your_score_1to5") in (None, ""):
            return
        corpus = row["corpus"].strip(); q = row["question"].strip(); ans = row["system_answer"]
        rec_src = idx.get((corpus, q, (ans or "")[:80])) or idx.get((corpus, q))
        item = {"question": q, "category": row.get("category", ""),
                "reference_answer": row.get("reference_answer", ""), "answer": ans,
                "retrieved": ([{"page": c.get("page"), "text": c.get("text")} for c in rec_src.get("retrieved_trim", [])]
                              if rec_src else []),
                "expected_pages": "", "expected_source": ""}
        try:
            jr = judge_openai(item, runs=runs)
        except Exception as e:
            print(f"  [ERR] {sid}: {type(e).__name__}: {str(e)[:100]}"); return
        rec = {"sample_id": sid, "corpus": corpus,
               "gpt_gen": jr["generation_avg"], "gpt_overall": jr["overall"],
               "had_context": bool(rec_src),
               "human1": float(row["your_score_1to5"]), "human2": float(a2[sid]["your_score_1to5"]),
               "haiku_overall": float(row["haiku_judge_overall"]) if row.get("haiku_judge_overall") else None,
               "sonnet_overall": float(row["sonnet_judge_overall"]) if row.get("sonnet_judge_overall") else None,
               "judge_cost": jr["judge_cost"]}
        with wlock:
            fh.write(json.dumps(rec) + "\n"); fh.flush()

    ids = list(a1.keys())
    print(f"[human] {len(ids)} sheet rows; judging those scored by both annotators")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(task, ids))
    fh.close()


def _pearson(a, b):
    a, b = np.array(a, float), np.array(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def analyze():
    rows = [json.loads(l) for l in CROSS_CKPT.read_text().splitlines()] if CROSS_CKPT.exists() else []
    out = {"model": OPENAI_MODEL, "n_records": len(rows), "corpora": {}, "agreement": {}, "human": {}}
    print(f"\n{'='*70}\nCROSS-FAMILY JUDGE: {OPENAI_MODEL} vs Haiku (n={len(rows)} records)\n{'='*70}")

    # ---- judge-vs-judge agreement (generation axis) ----
    g_gpt = [r["gen_gpt"] for r in rows if r.get("gen_haiku") is not None]
    g_hk = [r["gen_haiku"] for r in rows if r.get("gen_haiku") is not None]
    if g_gpt:
        out["agreement"] = {"n": len(g_gpt), "gen_pearson": _pearson(g_gpt, g_hk),
                            "gen_spearman": _spearman(g_gpt, g_hk),
                            "gpt_mean": float(np.mean(g_gpt)), "haiku_mean": float(np.mean(g_hk)),
                            "mean_offset_gpt_minus_haiku": float(np.mean(np.array(g_gpt) - np.array(g_hk)))}
        a = out["agreement"]
        print(f"\nGeneration-axis judge agreement (per record): Pearson r={a['gen_pearson']:.3f} "
              f"Spearman ρ={a['gen_spearman']:.3f}")
        print(f"  means: gpt-4o-mini {a['gpt_mean']:.2f} vs Haiku {a['haiku_mean']:.2f} "
              f"(offset {a['mean_offset_gpt_minus_haiku']:+.2f})")

    # ---- per corpus: EAR vs baseline under gpt judge, and ranking preservation ----
    by = defaultdict(lambda: defaultdict(dict))  # corpus -> variant -> qid -> row
    for r in rows:
        by[r["corpus"]][r["variant"]][r["qid"]] = r
    for corpus in ["wmp", "arxiv", "misalignment", "qasper"]:
        if corpus not in by:
            continue
        cv = by[corpus]
        means_gpt = {v: float(np.mean([x["gen_gpt"] for x in cv[v].values()])) for v in cv}
        means_hk = {v: float(np.mean([x["gen_haiku"] for x in cv[v].values()])) for v in cv}
        # EAR vs each baseline, paired Wilcoxon on gpt gen scores
        ear = cv.get("ear_full", {})
        cmp = {}
        for b in [x for x in CORE if x != "ear_full" and x in cv]:
            common = sorted(set(ear) & set(cv[b]))
            if not common:
                continue
            gp = stats.paired_wilcoxon([ear[q]["gen_gpt"] for q in common], [cv[b][q]["gen_gpt"] for q in common])
            hk = stats.paired_wilcoxon([ear[q]["gen_haiku"] for q in common], [cv[b][q]["gen_haiku"] for q in common])
            cmp[b] = {"gpt_delta": gp["mean_diff"], "gpt_p": gp["p"],
                      "haiku_delta": hk["mean_diff"], "haiku_p": hk["p"]}
        # ranking correlation between the two judges' per-system means
        vs = sorted(cv)
        rank_r = _spearman([means_gpt[v] for v in vs], [means_hk[v] for v in vs])
        out["corpora"][corpus] = {"means_gpt": means_gpt, "means_haiku": means_hk,
                                  "system_rank_spearman": rank_r, "ear_vs_baseline": cmp}
        print(f"\n[{corpus}] system-mean rank agreement (gpt vs haiku): Spearman ρ={rank_r:.3f}")
        print(f"  {'variant':16s} {'gpt':>5s} {'haiku':>6s}")
        for v in CORE:
            if v in means_gpt:
                print(f"  {v:16s} {means_gpt[v]:>5.2f} {means_hk[v]:>6.2f}")
        print(f"  EAR vs baseline (gen Δ, p) — gpt-4o-mini | Haiku:")
        for b, d in cmp.items():
            gsig = "✗" if d["gpt_p"] < 0.05 else "~"
            hsig = "✗" if d["haiku_p"] < 0.05 else "~"
            agree = "AGREE" if (d["gpt_p"] < 0.05) == (d["haiku_p"] < 0.05) else "DIFFER"
            print(f"    vs {b:16s} gpt Δ={d['gpt_delta']:+.3f} p={d['gpt_p']:.3f}{gsig} | "
                  f"haiku Δ={d['haiku_delta']:+.3f} p={d['haiku_p']:.3f}{hsig}  [{agree}]")

    # ---- human sample ----
    if HUMAN_CKPT.exists():
        hr = [json.loads(l) for l in HUMAN_CKPT.read_text().splitlines()]
        if hr:
            cons = [(r["human1"] + r["human2"]) / 2 for r in hr]
            gpt = [r["gpt_gen"] for r in hr]
            out["human"] = {"n": len(hr),
                            "gpt_vs_consensus_pearson": _pearson(gpt, cons),
                            "gpt_vs_consensus_spearman": _spearman(gpt, cons),
                            "gpt_vs_human1": _pearson(gpt, [r["human1"] for r in hr]),
                            "gpt_vs_human2": _pearson(gpt, [r["human2"] for r in hr]),
                            "gpt_vs_haiku": _pearson(gpt, [r["haiku_overall"] for r in hr if r.get("haiku_overall") is not None]) if all(r.get("haiku_overall") is not None for r in hr) else None,
                            "gpt_mean": float(np.mean(gpt)), "consensus_mean": float(np.mean(cons)),
                            "had_context_frac": float(np.mean([1 if r["had_context"] else 0 for r in hr]))}
            h = out["human"]
            print(f"\n[human sample] n={h['n']} (context recovered for {h['had_context_frac']*100:.0f}%)")
            print(f"  gpt-4o-mini vs human CONSENSUS: Pearson r={h['gpt_vs_consensus_pearson']:.3f} "
                  f"Spearman ρ={h['gpt_vs_consensus_spearman']:.3f}")
            print(f"  gpt-4o-mini vs annotator1 r={h['gpt_vs_human1']:.3f}; vs annotator2 r={h['gpt_vs_human2']:.3f}")
            print(f"  gpt-4o-mini mean {h['gpt_mean']:.2f} vs human consensus {h['consensus_mean']:.2f}")

    total_cost = sum(r.get("judge_cost", 0) for r in rows)
    if HUMAN_CKPT.exists():
        total_cost += sum(json.loads(l).get("judge_cost", 0) for l in HUMAN_CKPT.read_text().splitlines())
    out["total_cost_usd"] = round(total_cost, 4)
    print(f"\nTotal gpt-4o-mini judge cost: ${total_cost:.4f}")
    outpath = config.RESULTS_DIR / "crossjudge_gpt4omini.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"[out] {outpath}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="wmp,arxiv,misalignment,qasper")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--max-workers", type=int, default=6)
    args = ap.parse_args()
    if args.human:
        rejudge_human(max_workers=args.max_workers)
    elif args.analyze:
        analyze()
    else:
        rejudge_corpora([c.strip() for c in args.corpora.split(",")], runs=args.runs, max_workers=args.max_workers)
        rejudge_human(max_workers=args.max_workers)
        analyze()


if __name__ == "__main__":
    main()
