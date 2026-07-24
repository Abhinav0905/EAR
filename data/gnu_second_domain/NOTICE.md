# NOTICE — licensing for the GNU second-domain evaluation data

This folder contains a research evaluation built on three GNU manuals. Two different
licenses apply, by content type — please respect both.

## Original work in this folder — CC BY 4.0 (data) / MIT (code)

The following are **original work** of the EAR authors and are released under the same terms
as the rest of this artifact (data under CC BY 4.0, code under MIT):

- `gnu_manuals_questions.jsonl`, `gnu_manuals_questions.md`, `page_manifest.tsv`,
  `README_gnu_eval.md` — the hand-authored question set, reference answers, and page annotations.
- `summary.csv`, `per_question.csv`, `significance.csv`, `negatives.csv`,
  `harmonized_report.md`, `harmonized_report.json` — computed evaluation metrics (scores,
  costs, statistics). These contain **no** manual text.
- `package_gnu.py`, `README.md` — code and documentation.

## Embedded manual excerpts in the checkpoints — GFDL v1.3

`bash_checkpoint.jsonl`, `coreutils_checkpoint.jsonl`, and `make_checkpoint.jsonl` contain,
per record, **short verbatim excerpts** (≈300 characters × up to 6 chunks) retrieved from the
source manuals, plus model-generated answers that may quote them. These excerpts are portions of:

- **GNU Bash Reference Manual** — © Free Software Foundation, Inc.
- **GNU Coreutils Manual** — © Free Software Foundation, Inc.
- **GNU Make Manual** — © Free Software Foundation, Inc.

All three manuals are licensed under the **GNU Free Documentation License, Version 1.3**
(https://www.gnu.org/licenses/fdl-1.3.html), with no Invariant Sections. Those embedded
excerpts remain under the **GFDL** and are **not** relicensed by this artifact; they are
included only as retrieval context needed to reproduce and audit the evaluation.

## The manuals themselves are NOT redistributed here

The source PDFs are deliberately **not** included. Obtain them from the upstream GNU project:

- https://www.gnu.org/software/bash/manual/
- https://www.gnu.org/software/coreutils/manual/
- https://www.gnu.org/software/make/manual/

If you reuse the checkpoints, preserve this notice and the GFDL attribution for the excerpts.
