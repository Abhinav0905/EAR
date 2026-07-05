"""RAG pipelines. Each returns {answer, calls[], retrieved[], extra{}, wall_ms}.

`calls` lists only the benchmarked generation-model (Sonnet) invocations -> len(calls)
is the 'LLM calls' efficiency metric. EAR's flan-t5 SLM work is counted in extra.slm_calls.
"""
import time

from . import config, llm

_CE = None


def cross_encoder():
    global _CE
    if _CE is None:
        from sentence_transformers import CrossEncoder
        _CE = CrossEncoder(config.CROSS_ENCODER)
    return _CE


# ── retrieval helpers ─────────────────────────────────────────────────
def retrieve(vs, query, k):
    out = []
    for doc, dist in vs.similarity_search_with_score(query, k=k):
        out.append({"text": doc.page_content, "page": doc.metadata.get("page"),
                    "src": doc.metadata.get("src"), "score": float(dist)})
    return out


def ce_rerank(query, cands, top_n):
    if not cands:
        return cands
    scores = cross_encoder().predict([(query, c["text"]) for c in cands])
    for c, s in zip(cands, scores):
        c["rerank_score"] = float(s)
    return sorted(cands, key=lambda c: c["rerank_score"], reverse=True)[:top_n]


def pack_context(chunks, budget=None):
    budget = budget or config.TOKEN_BUDGET
    parts, used, seen = [], 0, set()
    for c in chunks:
        h = hash((c.get("text") or "")[:100])
        if h in seen:
            continue
        seen.add(h)
        t = len(c.get("text") or "") // 4
        if used + t > budget:
            break
        parts.append(f"[page {c.get('page')}] {c['text']}")
        used += t
    return "\n\n---\n\n".join(parts)


GEN_SYS = ("You are a precise assistant. Answer ONLY using the provided context. Cite page numbers "
           "like (page N) for factual claims. If the context is insufficient to answer, state that "
           "explicitly rather than guessing. Be concise and factual.")


def _gen(prompt, model, role="gen", system=None, max_tokens=600, temperature=0.0):
    r = llm.generate(prompt, model=model, system=system, temperature=temperature, max_tokens=max_tokens)
    rec = {"role": role, "model": r["model"], "input_tokens": r["input_tokens"],
           "output_tokens": r["output_tokens"], "cost": r["cost"], "latency_ms": r["latency_ms"]}
    return r["text"], rec


def _aprompt(q, ctx):
    return f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"


# ── 1. Simple RAG ─────────────────────────────────────────────────────
def simple_rag(vs, question, gen_model=None, k=None, **_):
    gen_model = gen_model or config.GEN_MODEL
    t0 = time.time()
    docs = retrieve(vs, question, k or config.TOP_K)
    ans, rec = _gen(_aprompt(question, pack_context(docs)), gen_model, system=GEN_SYS)
    return {"answer": ans, "calls": [rec], "retrieved": docs, "extra": {}, "wall_ms": (time.time() - t0) * 1000}


# ── 2. Reranked Simple RAG (same depth + cross-encoder as EAR; 1 LLM call) ──
def reranked_simple_rag(vs, question, gen_model=None, fetch=None, top_n=None, **_):
    gen_model = gen_model or config.GEN_MODEL
    t0 = time.time()
    cands = retrieve(vs, question, fetch or config.RERANK_FETCH)
    top = ce_rerank(question, cands, top_n or config.RERANK_TOP_N)
    ans, rec = _gen(_aprompt(question, pack_context(top)), gen_model, system=GEN_SYS)
    return {"answer": ans, "calls": [rec], "retrieved": top, "extra": {"reranked": True},
            "wall_ms": (time.time() - t0) * 1000}


# ── 3. Agentic RAG (Phase 2) ──────────────────────────────────────────
_AG_REWRITE = ("Rewrite the user question into 1-2 explicit, self-contained search queries. "
               "Output ONLY a JSON array of strings.\nQuestion: {q}")
_AG_COVER = ("Given the question and retrieved context, is the context sufficient to fully answer it?\n"
             "Question: {q}\nContext:\n{ctx}\n\n"
             'Output ONLY JSON: {{"coverage_score":0.0-1.0,"is_sufficient":true|false,"suggested_queries":["..."]}}')


def agentic_rag(vs, question, gen_model=None, max_iter=None, **_):
    gen_model = gen_model or config.GEN_MODEL
    max_iter = max_iter or config.MAX_AGENTIC_ITERATIONS
    t0 = time.time()
    calls, alld, seen = [], [], set()

    def add(docs):
        for d in docs:
            k = d["text"][:120]
            if k not in seen:
                seen.add(k); alld.append(d)

    add(retrieve(vs, question, config.TOP_K))
    it = 0
    for it in range(max_iter):
        rw, rec = _gen(_AG_REWRITE.format(q=question), gen_model, max_tokens=200); calls.append(rec)
        qs = llm.extract_json(rw)
        qs = qs if isinstance(qs, list) else ([qs] if isinstance(qs, str) else [question])
        for q in [str(x) for x in qs[:2] if x]:
            add(retrieve(vs, q, config.TOP_K))
        cv, rec = _gen(_AG_COVER.format(q=question, ctx=pack_context(alld)), gen_model, max_tokens=250); calls.append(rec)
        ce = llm.extract_dict(cv)
        if ce.get("is_sufficient") or float(ce.get("coverage_score", 0) or 0) >= config.COVERAGE_THRESHOLD:
            break
        for sq in (ce.get("suggested_queries") or [])[:1]:
            add(retrieve(vs, str(sq), config.TOP_K))
    ans, rec = _gen(_aprompt(question, pack_context(alld)), gen_model, system=GEN_SYS); calls.append(rec)
    return {"answer": ans, "calls": calls, "retrieved": alld[:config.RERANK_TOP_N + 3],
            "extra": {"iterations": it + 1}, "wall_ms": (time.time() - t0) * 1000}


# ── 4. EAR + LoRA (Phase 3B) with controller/ablation flags ───────────
def ear_lora(vs, question, slm, adapter_id, gen_model=None, *, use_rewrite=True, use_rerank=True,
             use_coverage=True, hybrid_ce=False, fetch=None, top_n=None, cov_threshold=None, max_iter=None, **_):
    gen_model = gen_model or config.GEN_MODEL
    fetch = fetch or config.RERANK_FETCH
    top_n = top_n or config.RERANK_TOP_N
    cov_threshold = config.COVERAGE_THRESHOLD if cov_threshold is None else cov_threshold
    max_iter = max_iter or config.MAX_EAR_ITERATIONS
    t0 = time.time()
    calls, alld, seen = [], [], set()
    slm_calls, slm_ms = 0, 0.0

    def add(docs):
        for d in docs:
            k = d["text"][:120]
            if k not in seen:
                seen.add(k); alld.append(d)

    final, it = [], 0
    for it in range(max_iter):
        queries = [question]
        if use_rewrite:
            ts = time.time()
            rq = slm.generate(adapter_id, f"rewrite query: {question}", max_len=64)
            slm_ms += (time.time() - ts) * 1000; slm_calls += 1
            if rq and len(rq.strip()) > 3:
                queries.append(rq.strip())
        for q in queries:
            add(retrieve(vs, q, fetch))
        cand = alld
        if use_rerank and cand:
            ts = time.time()
            outs = slm.generate_batch(
                adapter_id, [f"rank relevance: query: {question} document: {c['text'][:200]}" for c in cand], max_len=8)
            slm_ms += (time.time() - ts) * 1000; slm_calls += 1

            def score(o):
                o = (o or "").lower()
                return 1.0 if ("relevant" in o and "irrelevant" not in o) else (0.0 if "irrelevant" in o else 0.5)
            for c, o in zip(cand, outs):
                c["slm_score"] = score(o)
            ranked = sorted(cand, key=lambda c: c["slm_score"], reverse=True)
            if hybrid_ce:
                ranked = ce_rerank(question, ranked[:max(top_n * 2, top_n)], top_n)
            final = ranked[:top_n]
        else:
            final = cand[:top_n]
        if use_coverage:
            ts = time.time()
            summary = " | ".join(f"{c.get('page')}: {c['text'][:80]}" for c in final[:3])
            cov = slm.generate(adapter_id, f"assess coverage: query: {question} documents: {summary[:300]}", max_len=24)
            slm_ms += (time.time() - ts) * 1000; slm_calls += 1
            if any(w in cov.lower() for w in ["sufficient", "yes", "covered", "complete"]):
                break
        else:
            break
    ans, rec = _gen(_aprompt(question, pack_context(final)), gen_model, system=GEN_SYS); calls.append(rec)
    return {"answer": ans, "calls": calls, "retrieved": final,
            "extra": {"iterations": it + 1, "slm_calls": slm_calls, "slm_latency_ms": round(slm_ms, 1),
                      "adapter": adapter_id},
            "wall_ms": (time.time() - t0) * 1000}


# ── 5. Self-RAG (faithful re-implementation) ──────────────────────────
_SR_NEED = 'Does answering this require retrieving documents? Output ONLY JSON {{"retrieve":true|false}}.\nQuestion: {q}'
_SR_REL = ('List 0-based indices of passages RELEVANT to the question. Output ONLY JSON {{"relevant":[ints]}}.\n'
           'Question: {q}\nPassages:\n{ps}')
_SR_CRIT = ('Is every claim in the answer SUPPORTED by the context, and is the answer USEFUL? '
            'Output ONLY JSON {{"supported":true|false,"useful":true|false}}.\nQuestion: {q}\nContext:\n{ctx}\nAnswer: {a}')


def self_rag(vs, question, gen_model=None, k=None, **_):
    gen_model = gen_model or config.GEN_MODEL
    k = k or config.RERANK_FETCH
    t0 = time.time()
    calls = []
    need, rec = _gen(_SR_NEED.format(q=question), gen_model, max_tokens=40); calls.append(rec)
    docs = []
    if llm.extract_dict(need, {"retrieve": True}).get("retrieve", True):
        docs = retrieve(vs, question, k)
        ps = "\n".join(f"[{i}] (page {d['page']}) {d['text'][:300]}" for i, d in enumerate(docs))
        rel, rec = _gen(_SR_REL.format(q=question, ps=ps), gen_model, max_tokens=120); calls.append(rec)
        default_idx = list(range(min(config.RERANK_TOP_N, len(docs))))
        parsed = llm.extract_json(rel)
        if isinstance(parsed, dict):
            idxs = parsed.get("relevant", default_idx)
        elif isinstance(parsed, list):
            idxs = parsed
        else:
            idxs = default_idx
        docs = [docs[i] for i in idxs if isinstance(i, int) and 0 <= i < len(docs)][:config.RERANK_TOP_N] \
            or docs[:config.RERANK_TOP_N]
    ctx = pack_context(docs) if docs else "(no retrieval performed)"
    ans, rec = _gen(_aprompt(question, ctx), gen_model, system=GEN_SYS); calls.append(rec)
    crit, rec = _gen(_SR_CRIT.format(q=question, ctx=ctx, a=ans), gen_model, max_tokens=60); calls.append(rec)
    if llm.extract_dict(crit).get("supported") is False and docs:
        ans, rec = _gen(_aprompt(question, ctx) + "\nUse ONLY the context; if a fact is absent, say so.",
                        gen_model, system=GEN_SYS); calls.append(rec)
    return {"answer": ans, "calls": calls, "retrieved": docs, "extra": {"reimpl": "self-rag"},
            "wall_ms": (time.time() - t0) * 1000}


# ── 6. FLARE (faithful re-implementation) ─────────────────────────────
_FL_DRAFT = "Draft a brief answer to the question from your own knowledge (may be incomplete).\nQuestion: {q}"
_FL_QUERY = ("Given the question and a draft answer, write ONE search query for the facts most needing "
             "verification. Output ONLY the query text.\nQuestion: {q}\nDraft: {d}")


def flare(vs, question, gen_model=None, k=None, steps=2, **_):
    gen_model = gen_model or config.GEN_MODEL
    k = k or config.TOP_K
    t0 = time.time()
    calls, alld, seen = [], [], set()

    def add(docs):
        for d in docs:
            kk = d["text"][:120]
            if kk not in seen:
                seen.add(kk); alld.append(d)

    draft, rec = _gen(_FL_DRAFT.format(q=question), gen_model, max_tokens=200); calls.append(rec)
    add(retrieve(vs, f"{question} {draft}", k))
    for _ in range(max(0, steps - 1)):
        q2, rec = _gen(_FL_QUERY.format(q=question, d=draft), gen_model, max_tokens=60); calls.append(rec)
        add(retrieve(vs, q2.strip() or question, k))
    ans, rec = _gen(_aprompt(question, pack_context(alld)), gen_model, system=GEN_SYS); calls.append(rec)
    return {"answer": ans, "calls": calls, "retrieved": alld[:config.RERANK_TOP_N + 3],
            "extra": {"reimpl": "flare"}, "wall_ms": (time.time() - t0) * 1000}
