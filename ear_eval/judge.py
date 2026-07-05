"""Harmonized dual-axis LLM judge (Bedrock Haiku): 9 rubrics, 3 runs, mean ± std @ temp 0.

One rubric, one judge model, identical across every corpus and pipeline so all numbers
sit on the same scale (the Tier-1 'harmonize the rubric' fix).
"""
import numpy as np

from . import config, llm

RETRIEVAL_RUBRICS = ["source_relevance", "source_coverage", "citation_accuracy", "chunk_sufficiency"]
GENERATION_RUBRICS = ["faithfulness", "completeness", "correctness", "hallucination_score", "answer_relevance"]
ALL_RUBRICS = RETRIEVAL_RUBRICS + GENERATION_RUBRICS

JUDGE_SYSTEM = """You are a strict, calibrated expert evaluator for retrieval-augmented generation (RAG).
Score a system's answer on 9 rubrics, each an INTEGER 1-5 (5 = best). Be critical and consistent.

RETRIEVAL RUBRICS (judge the retrieved context/sources):
- source_relevance: are the retrieved sources relevant to the question?
- source_coverage: do the sources cover everything the question needs?
- citation_accuracy: do the answer's cited pages/sections match the expected sources (if any)?
- chunk_sufficiency: were enough of the right chunks retrieved to fully answer?

GENERATION RUBRICS (judge the answer text):
- faithfulness: every claim grounded in the retrieved context; no fabrication.
- completeness: addresses ALL parts of the question.
- correctness: factually correct relative to the reference answer.
- hallucination_score: 5 = nothing beyond the sources; 1 = heavy hallucination.
- answer_relevance: directly answers what was asked.

For NEGATIVE / unanswerable questions (the reference says the document lacks this information):
the IDEAL behaviour is to DECLINE or state the information is unavailable. Reward correct refusal
highly (correctness, faithfulness, hallucination_score, answer_relevance); a confident fabricated
answer to such a question MUST score low.

Output ONLY JSON:
{"retrieval_scores":{"source_relevance":int,"source_coverage":int,"citation_accuracy":int,"chunk_sufficiency":int},
 "generation_scores":{"faithfulness":int,"completeness":int,"correctness":int,"hallucination_score":int,"answer_relevance":int},
 "retrieval_rationale":"<=2 sentences","generation_rationale":"<=2 sentences"}"""


def _clamp(x):
    try:
        return max(1.0, min(5.0, float(x)))
    except (TypeError, ValueError):
        return 3.0


def _one_pass(item, model):
    chunks = item.get("retrieved", []) or []
    ctx = "\n\n---\n\n".join(
        f"[chunk {i+1}] (page {c.get('page')}) {(c.get('text') or '')[:600]}"
        for i, c in enumerate(chunks[:8])
    ) or "(no sources retrieved)"
    prompt = (
        f"Question: {item['question']}\n"
        f"Category: {item.get('category','')}\n"
        f"Reference answer (ideal): {item.get('reference_answer','')}\n"
        f"Expected sources/pages: {item.get('expected_pages','')} {item.get('expected_source','')}\n\n"
        f"Retrieved context:\n{ctx}\n\n"
        f"System answer:\n{item.get('answer','')}\n\n"
        f"Score all 9 rubrics now."
    )
    r = llm.generate(prompt, model=model, system=JUDGE_SYSTEM, temperature=0.0, max_tokens=700)
    sc = llm.extract_json(r["text"]) or {}
    return sc, r["cost"]


def judge(item, model=None, runs=None):
    """Judge one (question, answer, retrieved) item `runs` times; return mean±std per rubric and axis."""
    model = model or config.JUDGE_MODEL
    runs = runs or config.JUDGE_RUNS
    per_rubric = {k: [] for k in ALL_RUBRICS}
    ret_avgs, gen_avgs, rationales = [], [], []
    total_cost = 0.0
    for _ in range(runs):
        sc, c = _one_pass(item, model)
        total_cost += c
        rs = sc.get("retrieval_scores", {}) or {}
        gs = sc.get("generation_scores", {}) or {}
        rvals = [_clamp(rs.get(k, 3)) for k in RETRIEVAL_RUBRICS]
        gvals = [_clamp(gs.get(k, 3)) for k in GENERATION_RUBRICS]
        for k, v in zip(RETRIEVAL_RUBRICS, rvals):
            per_rubric[k].append(v)
        for k, v in zip(GENERATION_RUBRICS, gvals):
            per_rubric[k].append(v)
        ret_avgs.append(float(np.mean(rvals)))
        gen_avgs.append(float(np.mean(gvals)))
        rationales.append({"retrieval": sc.get("retrieval_rationale", ""),
                           "generation": sc.get("generation_rationale", "")})

    def _std(x):
        return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    return {
        "judge_model": model, "runs": runs,
        "retrieval_avg": float(np.mean(ret_avgs)), "retrieval_avg_std": _std(ret_avgs),
        "generation_avg": float(np.mean(gen_avgs)), "generation_avg_std": _std(gen_avgs),
        "overall": float((np.mean(ret_avgs) + np.mean(gen_avgs)) / 2),
        "rubric_means": {k: float(np.mean(v)) for k, v in per_rubric.items()},
        "rubric_stds": {k: _std(v) for k, v in per_rubric.items()},
        "judge_cost": total_cost,
        "rationales": rationales,
    }
