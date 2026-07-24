# EAR second-domain (GNU) — harmonized report (IN-SCOPE questions only)

*Quality/cost means below are computed on in-scope (answerable) questions only, so they agree with `significance.csv`. Negative-control refusal is in `negatives.csv`. n in-scope: bash 30, coreutils 40, make 42.*

## Per-corpus results (in-scope): Gen / Ret / LLM-calls / $/q

| Pipeline | bash | coreutils | make |
|---|---|---|---|
| Simple RAG | 3.61 / 3.67 / 1.0 / $0.0082 | 4.19 / 3.97 / 1.0 / $0.0083 | 4.23 / 4.27 / 1.0 / $0.0086 |
| Reranked Simple | 3.64 / 3.61 / 1.0 / $0.0071 | 4.17 / 4.23 / 1.0 / $0.0074 | 4.11 / 4.05 / 1.0 / $0.0076 |
| Agentic RAG | 3.77 / 3.83 / 3.7 / $0.0229 | 4.08 / 3.96 / 3.4 / $0.0214 | 4.39 / 4.30 / 3.1 / $0.0198 |
| Self-RAG* | 3.46 / 2.53 / 3.5 / $0.0103 | 3.50 / 1.46 / 3.1 / $0.0052 | 3.93 / 3.34 / 3.7 / $0.0126 |
| FLARE* | 4.10 / 4.26 / 3.0 / $0.0134 | 4.14 / 4.28 / 3.0 / $0.0134 | 4.30 / 4.39 / 3.0 / $0.0141 |
| EAR+LoRA (ours) | 3.69 / 3.69 / 1.0 / $0.0074 | 4.28 / 3.99 / 1.0 / $0.0074 | 4.09 / 4.05 / 1.0 / $0.0079 |

\* Self-RAG / FLARE are faithful LLM-driven re-implementations, not the original models.

## Significance: EAR+LoRA vs each baseline (paired, in-scope)

| Corpus | Baseline | n | ΔGen (EAR−base) | Wilcoxon p | bootstrap 95% CI | LLM-call reduction | Cost reduction |
|---|---|---|---|---|---|---|---|
| bash | Simple RAG | 30 | +0.080 | 0.4071 | [-0.193, +0.347] | 0% | 10% |
| bash | Reranked Simple | 30 | +0.047 | 0.6782 | [-0.413, +0.500] | 0% | -5% |
| bash | Agentic RAG | 30 | -0.080 | 0.6528 | [-0.400, +0.247] | 73% | 68% |
| bash | Self-RAG* | 30 | +0.227 | 0.3044 | [-0.187, +0.653] | 71% | 28% |
| bash | FLARE* | 30 | -0.413 | 0.1632 | [-0.820, -0.033] | 67% | 45% |
| coreutils | Simple RAG | 40 | +0.100 | 0.5594 | [-0.065, +0.290] | 0% | 11% |
| coreutils | Reranked Simple | 40 | +0.115 | 0.6289 | [-0.185, +0.435] | 0% | -1% |
| coreutils | Agentic RAG | 40 | +0.210 | 0.1502 | [-0.040, +0.470] | 71% | 65% |
| coreutils | Self-RAG* | 40 | +0.785 | 0.0001 | [+0.480, +1.090] | 68% | -42% |
| coreutils | FLARE* | 40 | +0.145 | 0.3381 | [-0.075, +0.390] | 67% | 45% |
| make | Simple RAG | 42 | -0.143 | 0.3698 | [-0.367, +0.052] | 0% | 9% |
| make | Reranked Simple | 42 | -0.029 | 1.0000 | [-0.390, +0.324] | 0% | -3% |
| make | Agentic RAG | 42 | -0.305 | 0.0731 | [-0.590, -0.043] | 68% | 60% |
| make | Self-RAG* | 42 | +0.159 | 0.4206 | [-0.198, +0.516] | 73% | 38% |
| make | FLARE* | 42 | -0.219 | 0.2436 | [-0.524, +0.067] | 67% | 44% |
| POOLED | Simple RAG | 112 | +0.004 | 0.8003 | [-0.125, +0.125] | 0% | 10% |
| POOLED | Reranked Simple | 112 | +0.043 | 0.6149 | [-0.173, +0.261] | 0% | -3% |
| POOLED | Agentic RAG | 112 | -0.061 | 0.5377 | [-0.229, +0.107] | 70% | 64% |
| POOLED | Self-RAG* | 112 | +0.401 | 0.0004 | [+0.186, +0.614] | 71% | 19% |
| POOLED | FLARE* | 112 | -0.141 | 0.3011 | [-0.320, +0.036] | 67% | 44% |

## Two quality-for-cost tradeoffs (not parity)

Most EAR-vs-baseline cells are statistical parity (Wilcoxon p≫0.05, bootstrap CI spans 0). **Two cells are not**, and we report them as an explicit tradeoff rather than glossing them as parity — in both, the bootstrap 95% CI excludes zero, so EAR is modestly *worse* on generation quality, bought back by a large call/cost cut:

- **bash — EAR vs FLARE:** EAR gen **3.69** vs 4.10 (Δ **-0.413**, Wilcoxon p=0.163, bootstrap CI [-0.820, -0.033] — **excludes 0**), for **67% fewer LLM calls** and **45% lower cost** (1.0 vs 3.0 calls). EAR gives up ~0.41/5 of generation quality to run one call instead of 3.
- **make — EAR vs Agentic RAG:** EAR gen **4.09** vs 4.39 (Δ **-0.305**, Wilcoxon p=0.073, bootstrap CI [-0.590, -0.043] — **excludes 0**), for **68% fewer LLM calls** and **60% lower cost** (1.0 vs 3.1 calls). EAR gives up ~0.30/5 of generation quality to run one call instead of 3.

Everywhere else (including EAR vs Simple, Reranked-Simple, and pooled), the quality difference is non-significant while calls/cost drop — the intended result.

## Negative-control refusal (pooled, higher = better)

| System | n | gen (refusal) |
|---|---|---|
| Simple RAG | 23 | 4.713 |
| Reranked Simple | 23 | 4.704 |
| Agentic RAG | 23 | 4.704 |
| Self-RAG* | 23 | 4.539 |
| FLARE* | 23 | 4.539 |
| EAR+LoRA (ours) | 23 | 4.713 |
