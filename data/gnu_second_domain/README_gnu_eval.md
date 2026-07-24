# EAR second-domain evaluation — GNU technical-documentation corpus

This is the ground-truth question set for running the EAR six-system harness on a **second domain** (software technical documentation), to test whether the answer-quality and cost results found on the compliance domain (California WMPs) generalise. It is built to mirror the WMP evaluation so results are directly comparable.

## What this set is (and is not)

It supports the claim **"EAR generalises from regulatory-compliance QA to a second, structurally different domain (software technical documentation)."** It does **not** support "works on all data types" — that would need open-domain and conversational corpora as well. Keep the paper's wording to the two domains actually evaluated: compliance + technical documentation.

## Structural parallel to the WMP run

The WMP evaluation used one domain (utility wildfire-mitigation-plan compliance) split across three utilities. This set uses one domain (GNU tool documentation) split across three manuals, so the same per-corpus and pooled analyses apply unchanged:

| Manual | corpus id | WMP analogue | PDF pages | in-scope | neg (cross-doc) | neg (out-of-scope) | total |
|---|---|---|---|---|---|---|---|
| GNU Bash Reference Manual | `bash` | PG&E | 214 | 30 | 4 | 3 | 37 |
| GNU Coreutils Manual | `coreutils` | SCE | 319 | 40 | 4 | 4 | 48 |
| GNU Make Manual | `make` | PacifiCorp | 229 | 42 | 4 | 4 | 50 |
| **Total** | | | | 112 | 12 | 11 | **135** |

Negative ratio: **23/135 = 17.0%** (WMP run: 35/210 = 16.7%).

## Files

- `gnu_manuals_questions.jsonl` — one record per question. Fields: `qid`, `corpus`, `question`, `reference_answer`, `expected_pages`, `question_type`, `difficulty`, `source_document`.
- `gnu_manuals_questions.md` — the same content as human-readable tables, for eyeball review.
- `page_manifest.tsv` — `qid → corpus → pages → question`, for quick spot-checking against the PDFs.

## Question types

- `lookup` — a single option/flag or specific fact (e.g., "how do you sort numerically").
- `concept` — a definition or behaviour (e.g., "what is a phony target").
- `negative_cross_doc` — asked against one manual but only answerable from a **different** GNU manual (e.g., the make corpus is asked about bash's `$RANDOM`). Correct behaviour is a grounded refusal / scope-out, not an answer. These directly stress EAR's coverage gate: overlapping vocabulary across the three manuals ("variable", "shell", "target") makes naive retrieval want to answer.
- `negative_out_of_scope` — not answerable from any of the three manuals (nginx, git, Kubernetes, SQL, PyTorch, etc.). Correct behaviour is a grounded refusal.

For both negative types, `expected_pages` is empty and `reference_answer` states the expected refusal.

## Page-number convention (important)

`expected_pages` are **PDF physical pages**, 1-based — the page index in the file, not the printed page number in the manual's footer. The two differ because of front matter (title, TOC): e.g., bash printed page 1 is PDF page 7. Every page was verified by extracting that PDF page's text (`pdftotext -layout`) and confirming the answer text appears on it. If the harness chunker records pages differently (e.g., 0-based, or printed numbers), apply a constant offset per manual — do not assume the printed number.

## Reference answers

Each reference answer is the correct content a good RAG response should contain, kept to 1–3 sentences and grounded on the cited page. It is a comparison target for the judge, not a gold string to match verbatim.

## How to plug into the harness

Point the corpus loader at the three PDFs (map `bash`→`gnu-bash-reference-manual.pdf`, `coreutils`→`gnu-coreutils-manual.pdf`, `make`→`gnu-make-manual.pdf`), chunk them the same way the WMP corpus was chunked, and feed `gnu_manuals_questions.jsonl` as the question set. The `corpus` field lets the per-corpus significance table be regenerated exactly as in the WMP run. Run all six systems against the same chunked corpus so the generator is shared, as before.

## Licensing note (for any public release)

The three manuals are distributed under the **GNU Free Documentation License**. Using them as a retrieval corpus for a research evaluation is fine, but if any artifact is released publicly (e.g., Zenodo), do not redistribute the manual PDFs themselves as "the dataset." The **questions, reference answers, and page annotations in these files are original work** and can be released under the same license as the rest of the EAR artifact; point users to the upstream GNU manuals for the source documents. This is separate from the WMP corpus, which is public regulatory filings.

## Caveats to state honestly in the paper

- All three manuals share a register (command-reference prose), so this is one *domain*, three sub-corpora — not three independent domains.
- The coverage-gate threshold (0.7) was tuned on compliance text. If EAR's retrieval-axis or refusal behaviour shifts on this corpus, report it rather than re-tuning silently; a documented shift is a finding, not a failure.
- Question counts per corpus (bash=30, coreutils=40, make=42 in-scope) are smaller than the WMP 70/corpus. Pooled n=112 is adequate for the effect sizes seen on WMP, but per-corpus cells have less power; expand any corpus if a per-corpus claim needs it.
