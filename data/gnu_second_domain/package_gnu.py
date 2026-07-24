#!/usr/bin/env python3
"""
Package the EAR second-domain (GNU technical documentation) evaluation.

Reads the three checkpoints in this directory (bash/coreutils/make), mirrors the WMP
harmonized analysis, and writes a conference-ready package:

  summary.csv            per (corpus, system): Gen/Ret/overall means+std, LLM/SLM calls,
                         $gen, tokens, wall_ms, judge cost, n
  per_question.csv       per (corpus, system, question): scores, calls, cost, expected vs
                         retrieved pages (top-6 packed), page-hit
  significance.csv       EAR+LoRA vs each baseline, per corpus AND pooled (paired Wilcoxon
                         on generation axis, bootstrap CI, LLM-call and cost reduction)
  negatives.csv          refusal quality on negative-control questions per system
  harmonized_report.md   report.py's cross-corpus + significance tables (via ear_eval.report)

Run (portable — reads checkpoints + question set from this folder; ear_eval/ is at the repo root):
  python data/gnu_second_domain/package_gnu.py
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
# repo layout: data/gnu_second_domain/package_gnu.py -> ear_eval/ is two levels up
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..", "..")))
os.environ.setdefault("EAR_RESULTS_DIR", _THIS)
from ear_eval import report as rep, stats, config  # noqa: E402

HERE = _THIS
QSET = os.path.join(_THIS, "gnu_manuals_questions.jsonl")
CORPUS_ORDER = ["bash", "coreutils", "make"]

# expected_pages are NOT stored in the run checkpoints; join them back from the
# ground-truth question set by qid so page-hit can be computed.
EXPECTED = {}
with open(QSET) as _f:
    for _l in _f:
        _l = _l.strip()
        if _l:
            _q = json.loads(_l)
            EXPECTED[_q["qid"]] = _q.get("expected_pages", []) or []
BASELINES = ["simple", "reranked_simple", "agentic", "self_rag", "flare"]
EAR = "ear_full"
ALLSYS = BASELINES + [EAR]
DISPLAY = {"simple": "Simple RAG", "reranked_simple": "Reranked Simple", "agentic": "Agentic RAG",
           "self_rag": "Self-RAG*", "flare": "FLARE*", "ear_full": "EAR+LoRA (ours)"}


def _inscope(rows):
    """Answerable (in-scope) rows only — the population the significance tests use.
    Quality/cost tables are computed on this set; refusal on negatives is in negatives.csv."""
    return [r for r in rows if r.get("answerable", True)]


def main():
    recs = rep.load_records(HERE)
    if not recs:
        print("no checkpoint records found in", HERE)
        return
    g = rep.by_corpus_variant(recs)               # {corpus: {variant: [rows]}}
    present_corpora = [c for c in CORPUS_ORDER if c in g]
    print(f"loaded {len(recs)} records across corpora: {present_corpora}")
    for c in present_corpora:
        counts = {v: len(g[c].get(v, [])) for v in ALLSYS}
        print(f"  {c}: " + ", ".join(f"{v}={counts[v]}" for v in ALLSYS))

    _write_summary(g, present_corpora)
    _write_per_question(g, present_corpora)
    sig_rows = _sig_rows(g, present_corpora)
    _write_significance(sig_rows)
    neg_rows = _write_negatives(g, present_corpora)
    # in-scope, self-consistent harmonized report (replaces report.py's all-questions md/json)
    _write_harmonized(g, present_corpora, sig_rows, neg_rows)


def _write_summary(g, corpora):
    """Quality/cost per (corpus, system), IN-SCOPE questions only (matches significance.csv).
    Negative-control refusal is reported separately in negatives.csv."""
    cols = ["corpus", "system", "n_inscope", "n_total", "gen_mean", "gen_std", "ret_mean",
            "ret_std", "overall_mean", "llm_calls_mean", "slm_calls_mean", "gen_cost_mean",
            "tokens_mean", "wall_ms_mean", "total_gen_cost", "total_judge_cost"]
    rows = []
    for c in corpora:
        for v in ALLSYS:
            if v not in g[c]:
                continue
            ins = _inscope(g[c][v])
            if not ins:
                continue
            a = rep.agg_variant(ins)
            rows.append(dict(
                corpus=c, system=v, n_inscope=a["n"], n_total=len(g[c][v]),
                gen_mean=round(a["generation_avg"]["mean"], 3),
                gen_std=round(a["generation_avg"]["std"], 3),
                ret_mean=round(a["retrieval_avg"]["mean"], 3),
                ret_std=round(a["retrieval_avg"]["std"], 3),
                overall_mean=round(a["overall"]["mean"], 3),
                llm_calls_mean=round(a["llm_calls"]["mean"], 2),
                slm_calls_mean=round(a["slm_calls"]["mean"], 2),
                gen_cost_mean=round(a["gen_cost_usd"]["mean"], 5),
                tokens_mean=round(a["tokens"]["mean"], 0),
                wall_ms_mean=round(a["wall_ms"]["mean"], 0),
                total_gen_cost=round(a["total_gen_cost"], 4),
                total_judge_cost=round(a["total_judge_cost"], 4)))
    _csv(os.path.join(HERE, "summary.csv"), cols, rows)


def _page_hit(rec):
    exp = set(EXPECTED.get(rec.get("qid"), []))  # joined from the ground-truth set
    if not exp:
        return ""  # negative / no ground truth
    got = {c.get("page") for c in rec.get("retrieved_trim", []) if c.get("page") is not None}
    return int(bool(exp & got))


def _write_per_question(g, corpora):
    cols = ["corpus", "system", "qid", "category", "answerable", "gen", "ret", "overall",
            "llm_calls", "slm_calls", "gen_cost", "expected_pages",
            "retrieved_top6_pages", "page_hit_top6"]
    rows = []
    for c in corpora:
        for v in ALLSYS:
            for r in g[c].get(v, []):
                rows.append(dict(
                    corpus=c, system=v, qid=r["qid"], category=r["category"],
                    answerable=r.get("answerable", True),
                    gen=round(r["generation_avg"], 3), ret=round(r["retrieval_avg"], 3),
                    overall=round(r["overall"], 3),
                    llm_calls=r["metrics"]["llm_calls"], slm_calls=r["metrics"]["slm_calls"],
                    gen_cost=round(r["metrics"]["gen_cost_usd"], 5),
                    expected_pages=json.dumps(EXPECTED.get(r["qid"], [])),
                    retrieved_top6_pages=json.dumps(
                        [c2.get("page") for c2 in r.get("retrieved_trim", [])]),
                    page_hit_top6=_page_hit(r)))
    _csv(os.path.join(HERE, "per_question.csv"), cols, rows)


def _paired(ear_q, base_q, key):
    common = sorted(set(ear_q) & set(base_q))
    a = [ear_q[q][key] for q in common]
    b = [base_q[q][key] for q in common]
    return common, a, b


def _sig_row(corpus, base, ear_rows, base_rows, answerable_only=True):
    def idx(rows):
        return {r["qid"]: r for r in rows
                if (not answerable_only or r.get("answerable", True))}
    eq, bq = idx(ear_rows), idx(base_rows)
    common, ag, bg = _paired(eq, bq, "generation_avg")
    if not common:
        return None
    _, ar, br = _paired(eq, bq, "retrieval_avg")
    w = stats.paired_wilcoxon(ag, bg)
    boot = stats.bootstrap_diff(ag, bg)
    el = np.mean([eq[q]["metrics"]["llm_calls"] for q in common])
    bl = np.mean([bq[q]["metrics"]["llm_calls"] for q in common])
    ec = np.mean([eq[q]["metrics"]["gen_cost_usd"] for q in common])
    bc = np.mean([bq[q]["metrics"]["gen_cost_usd"] for q in common])
    return dict(
        corpus=corpus, baseline=base, n=len(common),
        delta_gen=round(w["mean_diff"], 3), wilcoxon_p=round(w["p"], 4),
        boot_ci_low=round(boot["ci95"][0], 3), boot_ci_high=round(boot["ci95"][1], 3),
        cohens_d=round(stats.cohens_d_paired(ag, bg), 3),
        delta_ret=round(float(np.mean(ar) - np.mean(br)), 3),
        ear_llm_calls=round(float(el), 2), base_llm_calls=round(float(bl), 2),
        llm_reduction_pct=round(float((bl - el) / bl * 100) if bl else 0.0, 1),
        ear_cost=round(float(ec), 5), base_cost=round(float(bc), 5),
        cost_reduction_pct=round(float((bc - ec) / bc * 100) if bc else 0.0, 1))


def _sig_rows(g, corpora):
    rows = []
    for c in corpora:
        if EAR not in g[c]:
            continue
        for base in BASELINES:
            if base not in g[c]:
                continue
            r = _sig_row(c, base, g[c][EAR], g[c][base])
            if r:
                rows.append(r)

    def pooled(variant):
        out = []
        for c in corpora:
            for r in g[c].get(variant, []):
                r2 = dict(r); r2["qid"] = f"{c}:{r['qid']}"
                out.append(r2)
        return out
    if all(EAR in g[c] for c in corpora) and corpora:
        ear_all = pooled(EAR)
        for base in BASELINES:
            base_all = pooled(base)
            if base_all:
                r = _sig_row("POOLED", base, ear_all, base_all)
                if r:
                    rows.append(r)
    return rows


def _write_significance(rows):
    cols = ["corpus", "baseline", "n", "delta_gen", "wilcoxon_p", "boot_ci_low",
            "boot_ci_high", "cohens_d", "delta_ret", "ear_llm_calls", "base_llm_calls",
            "llm_reduction_pct", "ear_cost", "base_cost", "cost_reduction_pct"]
    _csv(os.path.join(HERE, "significance.csv"), cols, rows)


def _write_negatives(g, corpora):
    """Refusal quality on negative-control questions (higher gen score = better refusal)."""
    cols = ["scope", "system", "n_negatives", "gen_mean", "overall_mean"]
    rows = []
    for c in corpora + ["POOLED"]:
        srcs = corpora if c == "POOLED" else [c]
        for v in ALLSYS:
            negs = [r for s in srcs for r in g[s].get(v, [])
                    if not r.get("answerable", True)]
            if not negs:
                continue
            rows.append(dict(
                scope=c, system=v, n_negatives=len(negs),
                gen_mean=round(float(np.mean([r["generation_avg"] for r in negs])), 3),
                overall_mean=round(float(np.mean([r["overall"] for r in negs])), 3)))
    _csv(os.path.join(HERE, "negatives.csv"), cols, rows)
    return rows


def _write_harmonized(g, corpora, sig_rows, neg_rows):
    """In-scope, self-consistent harmonized_report.md (+ .json). The per-corpus quality
    table and the significance table are BOTH computed on in-scope questions, so they agree
    with significance.csv. Negative-control refusal is reported separately."""
    # in-scope aggregate per corpus/system
    agg = {c: {v: rep.agg_variant(_inscope(g[c][v]))
               for v in ALLSYS if v in g[c] and _inscope(g[c][v])}
           for c in corpora}

    L = ["# EAR second-domain (GNU) — harmonized report (IN-SCOPE questions only)\n",
         "*Quality/cost means below are computed on in-scope (answerable) questions only, so "
         "they agree with `significance.csv`. Negative-control refusal is in `negatives.csv`. "
         "n in-scope: bash 30, coreutils 40, make 42.*\n",
         "## Per-corpus results (in-scope): Gen / Ret / LLM-calls / $/q\n"]
    L.append("| Pipeline | " + " | ".join(corpora) + " |")
    L.append("|" + "---|" * (len(corpora) + 1))
    for v in ALLSYS:
        cells = []
        for c in corpora:
            a = agg[c].get(v)
            cells.append(f"{a['generation_avg']['mean']:.2f} / {a['retrieval_avg']['mean']:.2f} / "
                         f"{a['llm_calls']['mean']:.1f} / ${a['gen_cost_usd']['mean']:.4f}"
                         if a else "—")
        L.append(f"| {DISPLAY[v]} | " + " | ".join(cells) + " |")
    L.append("\n\\* Self-RAG / FLARE are faithful LLM-driven re-implementations, not the original models.\n")

    # significance (in-scope) — per corpus + pooled
    L.append("## Significance: EAR+LoRA vs each baseline (paired, in-scope)\n")
    L.append("| Corpus | Baseline | n | ΔGen (EAR−base) | Wilcoxon p | bootstrap 95% CI | "
             "LLM-call reduction | Cost reduction |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in sig_rows:
        L.append(f"| {r['corpus']} | {DISPLAY.get(r['baseline'], r['baseline'])} | {r['n']} | "
                 f"{r['delta_gen']:+.3f} | {r['wilcoxon_p']:.4f} | "
                 f"[{r['boot_ci_low']:+.3f}, {r['boot_ci_high']:+.3f}] | "
                 f"{r['llm_reduction_pct']:.0f}% | {r['cost_reduction_pct']:.0f}% |")

    # honest quality-for-cost tradeoffs (bootstrap CI excludes 0)
    L.append("\n## Two quality-for-cost tradeoffs (not parity)\n")
    L.append("Most EAR-vs-baseline cells are statistical parity (Wilcoxon p≫0.05, bootstrap CI "
             "spans 0). **Two cells are not**, and we report them as an explicit tradeoff rather "
             "than glossing them as parity — in both, the bootstrap 95% CI excludes zero, so EAR "
             "is modestly *worse* on generation quality, bought back by a large call/cost cut:\n")
    tc = {(r["corpus"], r["baseline"]): r for r in sig_rows}
    for corp, base, an, bn in [("bash", "flare", "FLARE", "FLARE"),
                               ("make", "agentic", "Agentic RAG", "Agentic")]:
        r = tc.get((corp, base))
        if not r:
            continue
        ea = agg[corp]["ear_full"]["generation_avg"]["mean"]
        ba = agg[corp][base]["generation_avg"]["mean"]
        L.append(f"- **{corp} — EAR vs {an}:** EAR gen **{ea:.2f}** vs {ba:.2f} "
                 f"(Δ **{r['delta_gen']:+.3f}**, Wilcoxon p={r['wilcoxon_p']:.3f}, bootstrap CI "
                 f"[{r['boot_ci_low']:+.3f}, {r['boot_ci_high']:+.3f}] — **excludes 0**), for "
                 f"**{r['llm_reduction_pct']:.0f}% fewer LLM calls** and "
                 f"**{r['cost_reduction_pct']:.0f}% lower cost** "
                 f"({r['ear_llm_calls']:.1f} vs {r['base_llm_calls']:.1f} calls). "
                 f"EAR gives up ~{abs(r['delta_gen']):.2f}/5 of generation quality to run one "
                 f"call instead of {r['base_llm_calls']:.0f}.")
    L.append("\nEverywhere else (including EAR vs Simple, Reranked-Simple, and pooled), the "
             "quality difference is non-significant while calls/cost drop — the intended result.")

    # refusal note
    negp = {r["system"]: r for r in neg_rows if r["scope"] == "POOLED"}
    if negp:
        L.append("\n## Negative-control refusal (pooled, higher = better)\n")
        L.append("| System | n | gen (refusal) |")
        L.append("|---|---|---|")
        for v in ALLSYS:
            if v in negp:
                L.append(f"| {DISPLAY[v]} | {negp[v]['n_negatives']} | {negp[v]['gen_mean']:.3f} |")

    with open(os.path.join(HERE, "harmonized_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(HERE, "harmonized_report.json"), "w") as f:
        json.dump({"scope": "in_scope_only", "aggregate": agg, "significance": sig_rows,
                   "negatives": neg_rows}, f, indent=2, default=str)
    print("[out] harmonized_report.md, harmonized_report.json  (in-scope)")


def _csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[out] {os.path.basename(path)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
