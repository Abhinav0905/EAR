# EAR second-domain evaluation — GNU technical documentation

Does the EAR cost/quality result from the compliance domain (California WMPs) **generalise to a
second, structurally different domain**? This package runs the same six-system harness, the same
generator, the same harmonized nine-rubric judge, and the same chunking/top-k on three GNU manuals
(bash / coreutils / make), using a hand-authored ground-truth question set built to mirror the WMP
run so the numbers are directly comparable.

**Bottom line: yes — the result reproduces, with two honest tradeoff cells.** Pooled over 112
in-scope questions, EAR+LoRA is statistically indistinguishable from every baseline on answer quality
(and significantly *beats* Self-RAG) while making **one** LLM call instead of 3–4 — cutting LLM calls
**67–71%** and cost **19–64%** vs the multi-call baselines. Per corpus, two cells are **not** parity
but an explicit quality-for-cost tradeoff: **bash EAR vs FLARE** (Δgen −0.41) and **make EAR vs
Agentic** (Δgen −0.31), in both of which the bootstrap 95% CI excludes zero — EAR gives up ~0.3–0.4/5
of generation quality to run one call instead of three. Negative-control refusal and evidence
retrieval hold up.

> All quality/cost tables in this package are computed on **in-scope (answerable) questions only**, so
> they agree with `significance.csv`; negative-control refusal is reported separately (`negatives.csv`).

## Setup (held constant, mirrors the WMP run)

| item | value |
|---|---|
| Corpora | `bash` (GNU Bash Reference, 214 pp / 622 chunks), `coreutils` (319 pp / 913), `make` (229 pp / ~700) — analogues of the WMP's PG&E/SCE/PacifiCorp |
| Questions | 135 hand-authored (`gnu_manuals_questions.jsonl`): **112 in-scope** + 12 cross-doc negatives + 11 out-of-scope negatives (17.0% negative, ≈ WMP's 16.7%) |
| Systems (6) | Simple RAG, Reranked Simple, Agentic RAG, Self-RAG\*, FLARE\*, **EAR+LoRA (ours)** |
| Generator | Claude Sonnet 4.5, one benchmarked call per query, shared across systems |
| Judge | harmonized 9-rubric LLM judge (Claude Haiku 4.5), ×3 @ temp 0 |
| Retrieval / chunking | Titan v2 embeddings, Chroma, CHUNK_SIZE 1200 / overlap 200, TOP_K 6, rerank fetch 20 |
| EAR | frozen flan-t5-small + LoRA (rank 8, q/v), coverage threshold 0.7 (tuned on WMP, **not** re-tuned) |
| Page convention | `expected_pages` are 1-based physical PDF pages; verified offset 0 vs harness chunk pages (108/112 token-match, spot-checks pass) |

\* Self-RAG / FLARE are faithful LLM-driven re-implementations, not the original trained models.

## Cross-corpus results (in-scope): Gen / Ret / LLM-calls / $ per query

| Pipeline | bash | coreutils | make |
|---|---|---|---|
| Simple RAG | 3.61 / 3.67 / 1.0 / $0.0082 | 4.19 / 3.97 / 1.0 / $0.0083 | 4.23 / 4.27 / 1.0 / $0.0086 |
| Reranked Simple | 3.64 / 3.61 / 1.0 / $0.0071 | 4.17 / 4.23 / 1.0 / $0.0074 | 4.11 / 4.05 / 1.0 / $0.0076 |
| Agentic RAG | 3.77 / 3.83 / 3.7 / $0.0229 | 4.08 / 3.96 / 3.4 / $0.0214 | 4.39 / 4.30 / 3.1 / $0.0198 |
| Self-RAG\* | 3.46 / 2.53 / 3.5 / $0.0103 | 3.50 / 1.46 / 3.1 / $0.0052 | 3.93 / 3.34 / 3.7 / $0.0126 |
| FLARE\* | 4.10 / 4.26 / 3.0 / $0.0134 | 4.14 / 4.28 / 3.0 / $0.0134 | 4.30 / 4.39 / 3.0 / $0.0141 |
| **EAR+LoRA (ours)** | **3.69 / 3.69 / 1.0 / $0.0074** | **4.28 / 3.99 / 1.0 / $0.0074** | **4.09 / 4.05 / 1.0 / $0.0079** |

Gen/Ret = generation- and retrieval-axis judge means (1–5), **in-scope questions only** (bash n=30,
coreutils n=40, make n=42) — so these agree with `significance.csv`. The gen values equal the paired
significance deltas exactly (e.g. bash EAR 3.69 − FLARE 4.10 = −0.41; make EAR 4.09 − Agentic 4.39 =
−0.31).

## Headline — pooled significance (EAR+LoRA vs each baseline, 112 in-scope, paired)

| Baseline | n | ΔGen (EAR−base) | Wilcoxon p | bootstrap 95% CI | LLM-call reduction | Cost reduction |
|---|---|---|---|---|---|---|
| Simple RAG | 112 | +0.004 | 0.80 | [−0.13, +0.13] | 0% | 10% |
| Reranked Simple | 112 | +0.043 | 0.61 | [−0.17, +0.26] | 0% | −3% |
| Agentic RAG | 112 | −0.061 | 0.54 | [−0.23, +0.11] | **70%** | **64%** |
| Self-RAG\* | 112 | **+0.401** | **0.0004** | [+0.19, +0.61] | **71%** | 19% |
| FLARE\* | 112 | −0.141 | 0.30 | [−0.32, +0.04] | **67%** | **44%** |

**Reading (pooled):** EAR is statistically **indistinguishable** from Simple, Reranked-Simple,
Agentic, and FLARE on answer quality (p ≫ 0.05, CIs span 0), and **significantly better** than
Self-RAG — while using **one** LLM call. Against the multi-call baselines that is a 67–71% call cut and
up to 64% cost cut. This is the WMP "quality maintained, cost/calls cut" result, reproduced on a
structurally different domain.

Pooled in-scope means (n=112): EAR **4.05** gen / 3.94 ret / **1.0** call / **$0.0076** — vs Simple
4.05 (1.0 call), Agentic 4.11 (3.4 calls, $0.0212), FLARE 4.19 (3.0 calls), Self-RAG 3.65.

### Two per-corpus cells are a quality-for-cost tradeoff, not parity

Pooled parity should not be over-read: **two per-corpus cells are genuine tradeoffs**, where the
bootstrap 95% CI on ΔGen excludes zero — EAR is modestly *worse* on generation quality, bought back by
a large call/cost cut. We report these honestly rather than as parity.

| Cell | EAR gen | baseline gen | ΔGen | Wilcoxon p | bootstrap CI | what EAR buys |
|---|---|---|---|---|---|---|
| **bash — EAR vs FLARE** | 3.69 | 4.10 | **−0.41** | 0.16 | [−0.82, −0.03] (excl. 0) | 67% fewer calls, 45% cheaper (1 vs 3) |
| **make — EAR vs Agentic** | 4.09 | 4.39 | **−0.31** | 0.07 | [−0.59, −0.04] (excl. 0) | 68% fewer calls, 60% cheaper (1 vs 3.1) |

In both, EAR trades ~0.3–0.4 of a point (on the 1–5 scale) of generation quality to collapse a
3-call pipeline into one. That is the intended engineering tradeoff — but it is a *tradeoff*, so we
state it as one. (Note the two effects wash out when pooled: EAR vs FLARE and EAR vs Agentic are both
non-significant across all 112 questions.)

## Negative-control refusal & evidence retrieval

- **Refusal (23 negatives, higher = better):** all systems refuse well; EAR **4.71** gen, tied for
  best — the coverage gate (threshold 0.7, tuned on compliance) transfers to technical docs without
  re-tuning.
- **Evidence retrieval, page-hit@top-6 (112 in-scope):** EAR **0.830** — on par with Simple 0.848,
  Reranked 0.839, Agentic 0.848, FLARE 0.866; Self-RAG lags at 0.339 (explaining its lower quality).

## Comparison to the WMP (first domain)

| | EAR gen | EAR calls | EAR $/q | EAR vs multi-call baselines |
|---|---|---|---|---|
| WMP (compliance) | 3.77 | 1.0 | $0.0072 | parity quality, ~70–80% fewer calls |
| GNU (technical docs) | 4.05 (in-scope) | 1.0 | $0.0076 | parity quality, 67–71% fewer calls |

The pattern is the same in both domains: EAR trades a multi-call agentic/iterative loop for a single
generation call plus cheap local SLM retrieval steps, holding judged quality constant. Absolute judge
scores run a bit higher on the GNU set (cleaner, more factual reference prose) but the *relative*
EAR-vs-baseline structure is unchanged.

## Files

| file | contents |
|---|---|
| `summary.csv` | per (corpus, system), **in-scope only**: Gen/Ret/overall mean+std, LLM/SLM calls, $gen, tokens, latency, total costs; `n_inscope` + `n_total` columns |
| `per_question.csv` | per (corpus, system, question): scores, calls, cost, expected vs top-6 retrieved pages, page-hit (all 810 incl. negatives, with `answerable` flag) |
| `significance.csv` | EAR vs each baseline, **per-corpus and pooled** (in-scope): ΔGen, Wilcoxon p, bootstrap CI, Cohen's d, LLM-call & cost reduction |
| `negatives.csv` | refusal quality on negative controls per system (per-corpus + pooled) |
| `harmonized_report.md` / `.json` | **in-scope, self-consistent** cross-corpus table + significance + the two quality-for-cost tradeoffs + refusal (agrees with `significance.csv`) |
| `{bash,coreutils,make}_checkpoint.jsonl` | full raw per-question records (resumable) |
| `package_gnu.py` | regenerates every CSV above from the checkpoints |
| `full_run.log`, `smoke_bash.log` | run logs |

Ground-truth question set lives at repo root: `gnu_manuals_questions.jsonl` / `.md`, `page_manifest.tsv`, `README_gnu_eval.md`.

## Reproduce

```bash
cd /Users/mac001/Documents/Patent/Patent_poc          # the ear_eval harness; needs Bedrock creds in WMP-CRIS/.env
export EAR_RESULTS_DIR=/Users/mac001/Documents/WMP-CRIS/evaluation/gnu_second_domain
for c in bash coreutils make; do python -m ear_eval.run --corpus $c --variants core; done   # checkpointed/resumable
python /Users/mac001/Documents/WMP-CRIS/evaluation/gnu_second_domain/package_gnu.py          # build CSVs + report
```

## Caveats (state honestly)

- **One domain, three sub-corpora.** All three manuals share a command-reference register, so this is
  a second *domain*, not three independent domains. The paper's claim is "compliance + technical
  documentation," not "all data types."
- **Per-corpus power.** In-scope n per corpus (30/40/42) is smaller than WMP's 70; the pooled n=112 is
  adequate for the observed effect sizes, but per-corpus cells have less power — expand a corpus if a
  per-corpus claim needs it.
- **Coverage threshold not re-tuned** (0.7, from compliance). It transferred cleanly (refusal held);
  a documented shift would be a finding, not a failure.
- **Judge caveat carries over.** As on WMP, absolute low-end judge scores are noisy; the claim rests on
  *relative* (paired) comparisons, which is what the significance table reports.
- **Licensing (GFDL).** The three manuals are under the GNU Free Documentation License. The questions,
  reference answers, and page annotations here are original work and releasable with the artifact, but
  **do not redistribute the manual PDFs as "the dataset"** — point users to the upstream GNU manuals.

## Cost

Generation $9.44 + judge $7.05 = **$16.49** on Bedrock (Sonnet gen + Haiku judge ×3), 810 records
(135 questions × 6 systems), plus ~$0.14 store-embedding + LoRA-event generation across the 3 corpora.
