"""Aggregate checkpoints -> harmonized cross-corpus tables, significance, ablations,
human-review doc, and a second-judge (Sonnet) agreement check.

  python -m ear_eval.report                 # tables + significance (no Bedrock)
  python -m ear_eval.report --human-eval 50 # blinded human-review doc + Sonnet judge agreement (Bedrock)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config, stats

BASELINES = ["simple", "reranked_simple", "agentic", "self_rag", "flare"]
EAR = "ear_full"
DISPLAY = {"simple": "Simple RAG", "reranked_simple": "Reranked Simple RAG", "agentic": "Agentic RAG",
           "self_rag": "Self-RAG*", "flare": "FLARE*", "ear_full": "EAR+LoRA (ours)"}


def load_records(results_dir=None):
    d = Path(results_dir or config.RESULTS_DIR)
    recs = []
    for f in sorted(d.glob("*_checkpoint.jsonl")):
        for line in f.read_text().splitlines():
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def _gen_suffix(gm):
    """Segregate any non-primary generator into its own corpus group so a generator-swap
    run (stored with the same corpus + variant names) never merges into the primary results.
    Primary generation is Sonnet (None = legacy Sonnet)."""
    if gm in (None, config.SONNET):
        return ""
    if gm == config.HAIKU:
        return "_haikuGen"
    return "_" + str(gm).split("/")[-1].split(":")[0].split(".")[-1][:16] + "Gen"


def by_corpus_variant(recs):
    g = defaultdict(lambda: defaultdict(list))
    for r in recs:
        # skip explicitly-marked reproducibility re-runs (Sonnet repeats) if any slip into the glob
        if r.get("run_type", "").startswith("sonnet_repeat"):
            continue
        ck = r["corpus"] + _gen_suffix(r.get("gen_model"))
        g[ck][r["variant"]].append(r)
    return g


def _ms(x):
    x = [v for v in x if v is not None]
    return {"mean": float(np.mean(x)) if x else 0.0,
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, "n": len(x)}


def agg_variant(rows):
    M = lambda k: [r["metrics"][k] for r in rows]
    return {
        "n": len(rows),
        "generation_avg": _ms([r["generation_avg"] for r in rows]),
        "retrieval_avg": _ms([r["retrieval_avg"] for r in rows]),
        "overall": _ms([r["overall"] for r in rows]),
        "llm_calls": _ms(M("llm_calls")), "slm_calls": _ms(M("slm_calls")),
        "gen_cost_usd": _ms(M("gen_cost_usd")), "wall_ms": _ms(M("wall_ms")), "tokens": _ms(M("tokens")),
        "total_gen_cost": float(np.sum(M("gen_cost_usd"))),
        "total_judge_cost": float(np.sum([r.get("judge_cost", 0.0) for r in rows])),
    }


def aggregate(g):
    return {c: {v: agg_variant(rows) for v, rows in vmap.items()} for c, vmap in g.items()}


def significance(g, ear=EAR):
    out = {}
    for corpus, vmap in g.items():
        if ear not in vmap:
            continue
        ear_q = {r["qid"]: r for r in vmap[ear]}
        out[corpus] = {}
        for base in BASELINES + ["ear_noadapter"]:
            if base not in vmap:
                continue
            base_q = {r["qid"]: r for r in vmap[base]}
            common = sorted(set(ear_q) & set(base_q))
            if not common:
                continue
            ag = [ear_q[q]["generation_avg"] for q in common]
            bg = [base_q[q]["generation_avg"] for q in common]
            ar = [ear_q[q]["retrieval_avg"] for q in common]
            br = [base_q[q]["retrieval_avg"] for q in common]
            el = np.mean([ear_q[q]["metrics"]["llm_calls"] for q in common])
            bl = np.mean([base_q[q]["metrics"]["llm_calls"] for q in common])
            ec = np.mean([ear_q[q]["metrics"]["gen_cost_usd"] for q in common])
            bc = np.mean([base_q[q]["metrics"]["gen_cost_usd"] for q in common])
            out[corpus][base] = {
                "n": len(common),
                "gen": {**stats.paired_wilcoxon(ag, bg), "bootstrap": stats.bootstrap_diff(ag, bg),
                        "cohens_d": stats.cohens_d_paired(ag, bg)},
                "ret": {**stats.paired_wilcoxon(ar, br), "bootstrap": stats.bootstrap_diff(ar, br)},
                "ear_llm_calls": float(el), "base_llm_calls": float(bl),
                "llm_reduction_pct": float((bl - el) / bl * 100) if bl else 0.0,
                "ear_cost": float(ec), "base_cost": float(bc),
                "cost_reduction_pct": float((bc - ec) / bc * 100) if bc else 0.0,
            }
    return out


def _fmt(ms):
    return f"{ms['mean']:.2f}±{ms['std']:.2f}"


def harmonized_table(agg):
    corpora = list(agg)
    lines = ["## Harmonized cross-corpus results (one rubric, Haiku judge ×3 @ temp0)\n"]
    header = "| Pipeline | " + " | ".join(
        f"{c}: Gen / Ret / LLMcalls / $" for c in corpora) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(corpora) + 1))
    order = [v for v in (BASELINES + [EAR]) if any(v in agg[c] for c in corpora)]
    for v in order:
        cells = []
        for c in corpora:
            a = agg[c].get(v)
            cells.append(f"{_fmt(a['generation_avg'])} / {_fmt(a['retrieval_avg'])} / "
                         f"{a['llm_calls']['mean']:.1f} / ${a['gen_cost_usd']['mean']:.4f}" if a else "—")
        lines.append(f"| {DISPLAY.get(v, v)} | " + " | ".join(cells) + " |")
    lines.append("\n\\* Self-RAG / FLARE are faithful re-implementations (LLM-driven), not the original trained models.")
    return "\n".join(lines)


def significance_table(sig):
    lines = ["\n## Significance: EAR+LoRA vs each baseline (paired, per-question)\n",
             "| Corpus | Baseline | n | ΔGen (EAR−base) | Wilcoxon p | bootstrap 95% CI | LLM-call reduction | Cost reduction |",
             "|---|---|---|---|---|---|---|---|"]
    for corpus, bmap in sig.items():
        for base, s in bmap.items():
            g = s["gen"]
            ci = s["gen"]["bootstrap"]["ci95"]
            lines.append(
                f"| {corpus} | {DISPLAY.get(base, base)} | {s['n']} | {g['mean_diff']:+.3f} | "
                f"{g['p']:.4f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                f"{s['llm_reduction_pct']:.0f}% | {s['cost_reduction_pct']:.0f}% |")
    lines.append("\nInterpretation: a non-significant ΔGen (p>0.05, CI spanning 0) with large negative "
                 "LLM-call/cost Δ is the target claim — quality maintained, cost/calls cut.")
    return "\n".join(lines)


def ablation_table(g):
    """Ablations compared on the COMMON question set per corpus (apples-to-apples)."""
    abls = ["ear_full", "ear_rank16", "ear_noadapter", "ear_norewrite", "ear_norerank",
            "ear_nocoverage", "ear_tau03", "ear_tau05", "ear_tau09"]
    out = ["\n## Ablations (aligned on common questions per corpus)\n"]
    for c, vmap in g.items():
        present = [v for v in abls if v in vmap and vmap[v]]
        if len(present) <= 1:
            continue
        common = set.intersection(*[set(r["qid"] for r in vmap[v]) for v in present])
        if not common:
            continue
        out.append(f"\n### {c}  (n={len(common)} common questions)\n| Variant | Gen | Ret | SLMcalls | $ |")
        out.append("|---|---|---|---|---|")
        for v in present:
            rs = [r for r in vmap[v] if r["qid"] in common]
            gm = np.mean([r["generation_avg"] for r in rs])
            rm = np.mean([r["retrieval_avg"] for r in rs])
            sc = np.mean([r["metrics"]["slm_calls"] for r in rs])
            co = np.mean([r["metrics"]["gen_cost_usd"] for r in rs])
            out.append(f"| {v} | {gm:.2f} | {rm:.2f} | {sc:.1f} | ${co:.4f} |")
    return "\n".join(out)


def qasper_f1_table(g):
    if "qasper" not in g:
        return ""
    vmap = g["qasper"]
    lines = ["\n## QASPER public benchmark — Answer-F1 (native metric) + harmonized judge\n",
             "| Pipeline | n | Answer-F1 | Gen (judge) | LLM calls | $/q |", "|---|---|---|---|---|---|"]
    for v in [x for x in (BASELINES + [EAR]) if x in vmap]:
        rs = vmap[v]
        f1 = np.mean([r["answer_f1"] for r in rs if r.get("answer_f1") is not None])
        gm = np.mean([r["generation_avg"] for r in rs])
        lc = np.mean([r["metrics"]["llm_calls"] for r in rs])
        co = np.mean([r["metrics"]["gen_cost_usd"] for r in rs])
        lines.append(f"| {DISPLAY.get(v, v)} | {len(rs)} | {f1:.3f} | {gm:.2f} | {lc:.1f} | ${co:.4f} |")
    return "\n".join(lines)


def main_report(results_dir=None):
    recs = load_records(results_dir)
    if not recs:
        print("No checkpoint records found in", config.RESULTS_DIR)
        return
    g = by_corpus_variant(recs)
    agg = aggregate(g)
    sig = significance(g)
    out_json = {"aggregate": agg, "significance": sig, "n_records": len(recs)}
    (config.RESULTS_DIR / "harmonized_report.json").write_text(json.dumps(out_json, indent=2, default=str))
    md = "\n\n".join([harmonized_table(agg), significance_table(sig),
                      qasper_f1_table(g), ablation_table(g)])
    (config.RESULTS_DIR / "harmonized_report.md").write_text(md)
    total_gen = sum(a["total_gen_cost"] for c in agg.values() for a in c.values())
    total_judge = sum(a["total_judge_cost"] for c in agg.values() for a in c.values())
    print(md)
    print(f"\n[cost] generation ${total_gen:.2f} + judge ${total_judge:.2f} = ${total_gen+total_judge:.2f}")
    print(f"[out] {config.RESULTS_DIR/'harmonized_report.md'}")


# ── Tier 2: human-review doc + second-judge (Sonnet) agreement ────────
def human_eval(k=50, do_judge_b=True, seed=0, results_dir=None):
    from . import judge as judgemod, llm
    recs = [r for r in load_records(results_dir) if "retrieved_trim" in r]
    if not recs:
        print("No records with retrieved context (re-run the full matrix to enable human-eval).")
        return
    rng = np.random.default_rng(seed)
    sample = [recs[i] for i in rng.choice(len(recs), size=min(k, len(recs)), replace=False)]
    rows, a_scores, b_scores = [], [], []
    for i, r in enumerate(sample, 1):
        item = {"question": r.get("question", ""), "category": r["category"],
                "reference_answer": r.get("reference_answer", ""),
                "expected_pages": r.get("expected_pages", []), "expected_source": "",
                "answer": r["answer"], "retrieved": r.get("retrieved_trim", [])}
        a_overall = r["overall"]
        b_overall = None
        if do_judge_b:
            jb = judgemod.judge(item, model=config.JUDGE_MODEL_B, runs=1)
            b_overall = jb["overall"]
            a_scores.append(a_overall); b_scores.append(b_overall)
        rows.append({"sample_id": f"H{i:03d}", "corpus": r["corpus"], "category": r["category"],
                     "question": r.get("question", ""), "reference_answer": r.get("reference_answer", ""),
                     "system_answer": r["answer"], "haiku_judge_overall": round(a_overall, 2),
                     "sonnet_judge_overall": round(b_overall, 2) if b_overall is not None else "",
                     "your_score_1to5": "", "your_notes": ""})
    # write blinded review CSV + markdown (pipeline identity hidden)
    import csv
    csv_path = config.RESULTS_DIR / "human_review.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    agree = {}
    if do_judge_b and a_scores:
        a, b = np.array(a_scores), np.array(b_scores)
        pear = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else 0.0
        agree = {"n": len(a), "pearson_r": pear, "mean_abs_diff": float(np.mean(np.abs(a - b))),
                 "haiku_mean": float(a.mean()), "sonnet_mean": float(b.mean())}
        (config.RESULTS_DIR / "judge_agreement.json").write_text(json.dumps(agree, indent=2))
    print(f"[human-eval] wrote {len(rows)} blinded items -> {csv_path}")
    if agree:
        print(f"[judge-agreement] Haiku vs Sonnet: r={agree['pearson_r']:.3f}, "
              f"mean|Δ|={agree['mean_abs_diff']:.3f} (n={agree['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-eval", type=int, default=0, help="sample size for blinded human-review doc")
    ap.add_argument("--no-judge-b", action="store_true")
    args = ap.parse_args()
    main_report()
    if args.human_eval:
        human_eval(k=args.human_eval, do_judge_b=not args.no_judge_b)


if __name__ == "__main__":
    main()
