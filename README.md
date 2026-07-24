# Embedded Agentic Retrieval (EAR) — Evaluation Dataset & Harness

Research artifact for the paper *Embedded Agentic Retrieval (EAR): Answer-Quality Parity With
Multi-Call Agentic RAG at a Single Generator Call* (under review, IEEE Access). This repository
ships the per-question evaluation **data** and the **code** that produced it, so every table in the
paper is reproducible.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21202098.svg)](https://doi.org/10.5281/zenodo.21202098)

**DOI:** [10.5281/zenodo.21202098](https://doi.org/10.5281/zenodo.21202098) (concept DOI — always resolves to the latest version). The specific v1.0.1 snapshot is [10.5281/zenodo.21202195](https://doi.org/10.5281/zenodo.21202195).

## What this is

EAR moves the retrieval control loop (query rewriting, reranking, coverage estimation) off the
metered generator and onto a small, LoRA-adapted language model beside the vector index, calling the
generator exactly once per query. This artifact contains the evidence: judge scores, costs, and call
counts for six pipelines across four corpora, plus cross-family robustness runs, ablations, a
multi-document experiment, and human validation.

**Second-domain generalization (v1.1).** [`data/gnu_second_domain/`](data/gnu_second_domain/) adds a
second, structurally different domain — **software technical documentation** (GNU Bash / Coreutils /
Make manuals, 135 hand-authored questions) — evaluated with the identical six-system harness, generator,
and nine-rubric judge. The compliance-domain result reproduces: pooled over 112 in-scope questions, EAR
holds answer-quality parity with the baselines (and beats Self-RAG) at one LLM call, cutting calls
67–71% and cost 19–64% vs the multi-call baselines, with two per-corpus cells reported honestly as a
quality-for-cost tradeoff. See its [`README.md`](data/gnu_second_domain/README.md).

## Repository structure

```
.
├── data/                # evaluation records (CC BY 4.0) — see data/README.md for full schema
│   ├── README.md        # per-file contents, record schema, run_type convention, correction note
│   ├── *_checkpoint.jsonl
│   ├── *_repeat.jsonl        # Sonnet temperature-0 reproducibility re-runs (see data/README.md)
│   ├── crossjudge_*         # cross-family (OpenAI GPT-4o-mini) judge re-scores
│   ├── *.json               # cross-family-judge and retriever-precision summaries
│   └── gnu_second_domain/   # second-domain study (GNU docs): CSVs, checkpoints, question set,
│                            #   package_gnu.py, NOTICE.md (checkpoint excerpts are GFDL — see NOTICE)
├── ear_eval/            # evaluation harness (MIT) — the Python code that produced data/
├── CITATION.cff
├── .zenodo.json         # Zenodo archival metadata
├── LICENSE              # MIT (covers ear_eval/ code)
└── data/LICENSE         # CC BY 4.0 (covers data/)
```

## Data

See [`data/README.md`](data/README.md) for the complete file list, record schema, the `gen_model`
legend, the `run_type` convention, and the generator-swap correction note. Question text and
reference answers are embedded inline in every record, so the dataset is self-contained (QASPER uses
its native human-authored questions). No PII; scores are numeric.

## Code (harness)

`ear_eval/` is a Python package (built for Amazon Bedrock generation/judging + local Chroma +
Flan-T5-Small/LoRA). Key modules: `run.py` (orchestrator), `pipelines.py` (the six RAG pipelines),
`slm.py` (EAR's Flan-T5 + LoRA controller), `judge.py` (nine-rubric LLM judge),
`judge_openai.py` (cross-family GPT-4o-mini judge), `crossgen.py` (cross-family generation),
`multidoc*.py` (combined-index experiments), `report.py` (aggregation/tables), `stats.py`
(paired Wilcoxon, bootstrap, TOST).

Reproduce, e.g.:
```bash
pip install -r requirements.txt          # (torch, peft, transformers, langchain-chroma, boto3, openai, scipy, ...)
python -m ear_eval.run --corpus wmp --variants core        # primary results
python -m ear_eval.report                                  # aggregate into tables

# second domain (GNU technical docs); needs the manual PDFs (not redistributed — see NOTICE.md)
for c in bash coreutils make; do python -m ear_eval.run --corpus $c --variants core; done
python data/gnu_second_domain/package_gnu.py               # rebuild the GNU CSVs + report
```
Credentials (AWS Bedrock, OpenAI) are read from a `.env` file whose path is set by `EAR_ENV_PATH`;
output/result paths are overridable via `EAR_RESULTS_DIR` / `EAR_WORK_DIR` (defaults in
`ear_eval/config.py`). No credentials are included in this repository.

## Reproducibility notes

- Answer-quality scores come from an LLM judge (Claude Haiku 4.5, nine rubrics, ×3 at temperature 0);
  QASPER additionally has token-level Answer-F1. A cross-family judge (OpenAI GPT-4o-mini) and a
  cross-family generator (OpenAI GPT-4o-mini) reproduce the parity conclusions.
- The `*_repeat.jsonl` files are Sonnet temperature-0 reproducibility re-runs, not a separate model
  condition; do not pool them with the primary checkpoints. See `data/README.md`.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository" button).
After the DOI is minted, cite the Zenodo record. BibTeX will be generated by Zenodo.

## License

Dual-licensed: **code** (`ear_eval/`) under the **MIT License** ([`LICENSE`](LICENSE)); **data**
(`data/`) under **CC BY 4.0** ([`data/LICENSE`](data/LICENSE)).

**Exception — GNU second-domain checkpoints:** the three `data/gnu_second_domain/*_checkpoint.jsonl`
files embed short verbatim excerpts of the GNU Bash/Coreutils/Make manuals (© FSF), which remain under
the **GFDL v1.3** and are not relicensed here. The manuals themselves are not redistributed. See
[`data/gnu_second_domain/NOTICE.md`](data/gnu_second_domain/NOTICE.md).

## Archiving (Zenodo DOI)

This repo is archived on Zenodo via the GitHub↔Zenodo integration: each published GitHub Release is
archived and versioned under a stable **concept DOI** ([10.5281/zenodo.21202098](https://doi.org/10.5281/zenodo.21202098))
that always resolves to the latest version. The current archived snapshot is **v1.0.1**
([10.5281/zenodo.21202195](https://doi.org/10.5281/zenodo.21202195)). Future releases add new versions
under the same concept DOI.

## Related

- Paper: *Embedded Agentic Retrieval (EAR)…*, under review at IEEE Access.
- Patent: US Provisional Application No. 63/967,742.

## Contact

Abhinav Kumar ([ORCID 0009-0009-1839-841X](https://orcid.org/0009-0009-1839-841X)) — rushtoabhinavin@gmail.com (personal, for continuity) · abhinav@aidash.com (AiDash).
