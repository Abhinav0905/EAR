"""Corpora: PDF -> Titan/Chroma store (cached) + grounded question generation (Haiku).

Questions are generated per-chunk so we know the exact supporting page(s) -> gives real
expected-page labels for the citation_accuracy / source rubrics. Negatives are verified
unanswerable against the document before inclusion.
"""
import json
import shutil
import numpy as np

from . import config, llm

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import re


# ── PDF extraction / chunking ─────────────────────────────────────────
def extract_pdf_pages(pdf_path):
    import fitz
    doc = fitz.open(str(pdf_path))
    return [(i + 1, pg.get_text("text")) for i, pg in enumerate(doc)]


def find_references_cutoff(pages):
    n = len(pages)
    for idx in range(max(1, int(n * 0.55)), n):
        _, txt = pages[idx]
        if re.search(r"(?m)^\s*references\s*$", txt, re.I):
            return idx
        if len(re.findall(r"(?m)^\s*\[\d+\]\s", txt)) >= 5:
            return idx
    return None


def chunk_pages(pages, chunk_size, overlap, trim_refs):
    cut = find_references_cutoff(pages) if trim_refs else None
    use = pages if cut is None else pages[:cut]
    sp = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap,
                                        separators=["\n\n", "\n", ". ", " ", ""])
    chunks = []
    for page_no, text in use:
        if len(text.strip()) < 40:
            continue
        for piece in sp.split_text(text):
            if len(piece.strip()) < 40:
                continue
            chunks.append({"text": piece, "page": page_no, "idx": len(chunks)})
    return chunks, cut


# ── Vector store (cached / resumable) ─────────────────────────────────
def build_store(corpus_name, pdf_path, chunk_size=None, overlap=None, trim_refs=True, rebuild=False):
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    persist = config.WORK_DIR / "chroma" / corpus_name
    meta_path = config.WORK_DIR / f"{corpus_name}_chunks.json"
    emb = llm.TitanEmbeddings()

    if not rebuild and persist.exists() and meta_path.exists():
        chunks = json.loads(meta_path.read_text())
        vs = Chroma(persist_directory=str(persist), collection_name=corpus_name, embedding_function=emb)
        try:
            cnt = vs._collection.count()
        except Exception:
            cnt = len(chunks)
        if cnt >= len(chunks) > 0:
            print(f"[corpus:{corpus_name}] reusing cached store ({cnt} vectors)")
            return vs, chunks
    if persist.exists():
        shutil.rmtree(persist)

    pages = extract_pdf_pages(pdf_path)
    chunks, cut = chunk_pages(pages, chunk_size, overlap, trim_refs)
    print(f"[corpus:{corpus_name}] {len(chunks)} chunks from {len(pages)} pages "
          f"(refs trimmed at page {cut+1 if cut is not None else 'n/a'}); embedding with Titan...")
    docs = [Document(page_content=c["text"],
                     metadata={"doc_id": corpus_name, "page": c["page"], "idx": c["idx"]})
            for c in chunks]
    vs = Chroma.from_documents(documents=docs, embedding=emb,
                              persist_directory=str(persist), collection_name=corpus_name)
    meta_path.write_text(json.dumps(chunks))
    print(f"[corpus:{corpus_name}] stored {len(docs)} vectors at {persist}")
    return vs, chunks


# ── Question generation (grounded per-chunk + verified negatives) ─────
_FLAVOR = {
    "simple_factual": "specific single-fact",
    "method_detail": "technical method/numeric-detail (a setting, metric, mechanism, or number)",
    "multi_hop": "multi-hop reasoning",
    "cross_section_synthesis": "cross-section synthesis (compare/summarize across parts)",
}

GEN_SINGLE = """Write ONE evaluation question answerable ONLY from the text chunk below, taken from the document "{title}".
Make it a {flavor} question; provide a concise correct reference_answer (1-3 sentences). Avoid trivial or ambiguous questions; ask about the content directly (do not say "the chunk").
CHUNK (page {page}):
{chunk}
Output ONLY JSON: {{"question":"...","reference_answer":"..."}}"""

GEN_MULTI = """Write ONE {flavor} evaluation question about the document "{title}" whose answer REQUIRES combining information from at least two of the excerpts below. Provide a concise correct reference_answer.
EXCERPTS:
{excerpts}
Output ONLY JSON: {{"question":"...","reference_answer":"..."}}"""

NEG_GEN = """You are creating NEGATIVE-CONTROL questions for a RAG eval on the document "{title}" (summary/excerpts below).
Generate EXACTLY {n} questions that are NOT answerable from this document. Prefer information TYPES the document does not contain (real-world dollar cost, energy/carbon, wall-clock runtime in hours on named hardware, programming language, licensing, user studies, comparisons to methods it never mentions, applications/domains outside scope). AVOID asking for specific numbers/settings that might appear in tables.
DOCUMENT CONTEXT:
{context}
Output ONLY JSON: {{"questions":[{{"question":"...","category":"negative_out_of_scope"}}]}}"""

NEG_VERIFY = """Can the QUESTION be answered using ONLY the document below (including tables/appendices)?
DOCUMENT:
{context}
QUESTION: {question}
Output ONLY JSON: {{"answerable": true|false}}"""


def _gen_json(prompt, model, max_tokens=600):
    r = llm.generate(prompt, model=model, temperature=0.5, max_tokens=max_tokens)
    return llm.extract_json(r["text"]), r["cost"]


def _page_for_quote(quote, chunks):
    if not quote:
        return []
    q = quote.strip()[:60].lower()
    for c in chunks:
        if q and q in (c["text"] or "").lower():
            return [c["page"]]
    return []


def generate_questions(corpus_name, chunks, title, n, aux_model=None, seed=0, cache=True):
    aux_model = aux_model or config.AUX_MODEL
    cache_path = config.RESULTS_DIR / f"{corpus_name}_questions.json"
    existing = []
    if cache and cache_path.exists():
        data = json.loads(cache_path.read_text())
        existing = data.get("questions", data) if isinstance(data, dict) else data
    # Negative-aware top-up: fill the deficit toward ~25% negatives + n total.
    existing_neg = sum(1 for q in existing if str(q.get("category", "")).startswith("negative"))
    existing_ans = len(existing) - existing_neg
    target_neg = round(0.25 * n) if n >= 4 else (1 if n >= 2 else 0)
    n_neg = max(0, target_neg - existing_neg)
    rem = max(0, (n - target_neg) - existing_ans)
    if n_neg == 0 and rem == 0:
        print(f"[corpus:{corpus_name}] reusing cached {len(existing)} questions")
        return existing

    rng = np.random.default_rng(seed + len(existing))
    n_multi = round(rem * 0.30) if rem >= 4 else 0
    n_single = rem - n_multi

    # spread sampled chunks across the document
    order = sorted(range(len(chunks)), key=lambda i: chunks[i]["page"])

    def pick(k):
        k = min(max(0, k), len(order))
        if k <= 0:
            return []
        return [chunks[order[i]] for i in np.unique(np.linspace(0, len(order) - 1, num=k).astype(int))]

    single_cats = ["simple_factual", "method_detail"]
    multi_cats = ["multi_hop", "cross_section_synthesis"]
    out, cost = [], 0.0

    print(f"[corpus:{corpus_name}] generating {n_single} single + {n_multi} multi + {n_neg} negative questions...")
    n_single_have = 0
    for j, ch in enumerate(pick(n_single * 3)):
        if n_single_have >= n_single:
            break
        cat = single_cats[j % 2]
        js, c = _gen_json(GEN_SINGLE.format(title=title, flavor=_FLAVOR[cat], page=ch["page"], chunk=ch["text"][:1400]), aux_model)
        cost += c
        if js and js.get("question") and js.get("reference_answer"):
            out.append({"question": js["question"].strip(), "reference_answer": js["reference_answer"].strip(),
                        "category": cat, "expected_pages": [ch["page"]], "expected_source": "", "answerable": True})
            n_single_have += 1

    multi_pool = pick(min(len(chunks), max(6, n_multi * 3)))
    n_multi_have = 0
    for j in range(n_multi * 3):
        if n_multi_have >= n_multi or len(multi_pool) < 2:
            break
        sel = list(rng.choice(len(multi_pool), size=min(3, len(multi_pool)), replace=False))
        exc = [multi_pool[s] for s in sel]
        cat = multi_cats[j % 2]
        excerpts = "\n\n".join(f"(page {e['page']}) {e['text'][:600]}" for e in exc)
        js, c = _gen_json(GEN_MULTI.format(title=title, flavor=_FLAVOR[cat], excerpts=excerpts), aux_model)
        cost += c
        if js and js.get("question") and js.get("reference_answer"):
            out.append({"question": js["question"].strip(), "reference_answer": js["reference_answer"].strip(),
                        "category": cat, "expected_pages": sorted({e["page"] for e in exc}),
                        "expected_source": "", "answerable": True})
            n_multi_have += 1

    # negatives: generate in SMALL batches (avoid JSON truncation), verify each unanswerable
    context = "\n\n".join(c["text"][:500] for c in pick(20))[:60000]
    kept, attempts, seen_q = 0, 0, set()
    while kept < n_neg and attempts < max(4, n_neg):
        attempts += 1
        batch = max(6, min(10, (n_neg - kept) * 2))
        raw, c = _gen_json(NEG_GEN.format(title=title, n=batch, context=context), aux_model, max_tokens=1500)
        cost += c
        cands = raw.get("questions", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
        for q in cands:
            if kept >= n_neg:
                break
            qq = ((q.get("question") if isinstance(q, dict) else str(q)) or "").strip()
            if not qq or qq.lower() in seen_q:
                continue
            seen_q.add(qq.lower())
            v, c2 = _gen_json(NEG_VERIFY.format(context=context, question=qq), aux_model, max_tokens=120)
            cost += c2
            if isinstance(v, dict) and v.get("answerable") is False:
                cat = q.get("category", "negative_out_of_scope") if isinstance(q, dict) else "negative_out_of_scope"
                out.append({"question": qq, "reference_answer": "The document does not provide this information.",
                            "category": cat, "expected_pages": [], "expected_source": "", "answerable": False})
                kept += 1

    start = len(existing)
    for i, q in enumerate(out, 1):
        q["id"] = f"{corpus_name}_q{start + i}"
    combined = existing + out
    print(f"[corpus:{corpus_name}] produced {len(out)} new (total {len(combined)}) questions (gen cost ${cost:.3f})")
    if cache:
        cache_path.write_text(json.dumps({"corpus": corpus_name, "title": title, "questions": combined}, indent=2))
    return combined


def load_gnu_questions(path, corpus_name):
    """Load the provided GNU second-domain question set, filtered to one corpus.

    The file (gnu_manuals_questions.jsonl) is hand-authored ground truth with fields
    qid/corpus/question/reference_answer/expected_pages/question_type/difficulty. Pages are
    1-based physical PDF pages, matching this harness's chunk .page (verified offset 0), so
    they are used as-is. question_type -> category; negatives are answerable=False.
    """
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if q.get("corpus") != corpus_name:
                continue
            cat = q.get("question_type", "lookup")
            out.append({
                "id": q["qid"],
                "question": q["question"],
                "reference_answer": q.get("reference_answer", ""),
                "category": cat,
                "expected_pages": q.get("expected_pages", []) or [],
                "expected_source": "",
                "answerable": not str(cat).startswith("negative"),
                "curated": True,
            })
    if not out:
        raise ValueError(f"no questions found for corpus {corpus_name!r} in {path}")
    return out


def load_wmp_golden(path, corpus_name="wmp"):
    """Map the curated WMP golden set into the harness schema."""
    data = json.loads(open(path).read())
    qs = data["questions"] if isinstance(data, dict) and "questions" in data else data
    out = []
    for i, q in enumerate(qs, 1):
        cat = q.get("category", "simple_factual")
        out.append({
            "id": q.get("id", f"{corpus_name}_g{i}"),
            "question": q["question"],
            "reference_answer": q.get("expected_answer", ""),
            "category": cat,
            "expected_pages": q.get("expected_page_numbers", []),
            "expected_source": ", ".join(q.get("expected_source_sections", []) or []),
            "answerable": not str(cat).startswith("negative"),
            "curated": True,
        })
    return out
