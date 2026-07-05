"""EAR retrieval-side SLM: frozen flan-t5-small + PEFT LoRA, with batched inference.

Supports the LoRA-capacity ablation (rank/targets configurable, plus a no-adapter mode
that runs the frozen base model) and Bedrock-Haiku-generated micro-training events.
"""
import os
import time
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from . import config, llm

DEVICE = os.environ.get(
    "EAR_SLM_DEVICE",
    "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
)


class _EventDS(Dataset):
    def __init__(self, events, tok, max_in=512, max_out=64):
        self.e, self.tok, self.mi, self.mo = events, tok, max_in, max_out

    def __len__(self):
        return len(self.e)

    def __getitem__(self, i):
        e = self.e[i]
        x = self.tok(e["input_text"], max_length=self.mi, padding="max_length", truncation=True, return_tensors="pt")
        y = self.tok(e["target_text"], max_length=self.mo, padding="max_length", truncation=True, return_tensors="pt")
        lab = y["input_ids"].squeeze()
        lab[lab == self.tok.pad_token_id] = -100
        return {"input_ids": x["input_ids"].squeeze(), "attention_mask": x["attention_mask"].squeeze(), "labels": lab}


class EARSlm:
    def __init__(self, rank=None, alpha=None, dropout=None, targets=None, no_adapter=False, name="default"):
        self.no_adapter = no_adapter
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(config.SLM_MODEL_NAME)
        self.base = AutoModelForSeq2SeqLM.from_pretrained(config.SLM_MODEL_NAME).to(DEVICE)
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()
        self.peft_models = {}
        self._trainable = {}
        self.lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=rank or config.LORA_R, lora_alpha=alpha or config.LORA_ALPHA,
            lora_dropout=dropout if dropout is not None else config.LORA_DROPOUT,
            target_modules=targets or config.LORA_TARGET_MODULES, bias="none",
        )

    def base_param_count(self):
        return sum(p.numel() for p in self.base.parameters())

    def create_adapter(self, aid):
        m = get_peft_model(self.base, self.lora_cfg)
        self.peft_models[aid] = m
        self._trainable[aid] = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return self._trainable[aid]

    def adapter_param_count(self, aid):
        return self._trainable.get(aid, 0)

    def _model(self, aid):
        if self.no_adapter or aid is None:
            return self.base
        return self.peft_models[aid]

    def train(self, aid, events, epochs=15, lr=1e-4, max_steps=120, batch=4):
        if not events:
            return {"steps": 0, "avg_loss": None}
        m = self.peft_models[aid]
        m.train()
        dl = DataLoader(_EventDS(events, self.tok), batch_size=batch, shuffle=True)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
        total = min(max_steps, len(dl) * epochs)
        sch = get_linear_schedule_with_warmup(opt, max(1, total // 10), total)
        step, tot = 0, 0.0
        t0 = time.time()
        for _ in range(epochs):
            for b in dl:
                if step >= max_steps:
                    break
                b = {k: v.to(DEVICE) for k, v in b.items()}
                loss = m(**b).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
                tot += loss.item(); step += 1
                if step % 40 == 0:
                    print(f"    [slm:{self.name}] train step {step}/{max_steps} loss={loss.item():.3f}", flush=True)
            if step >= max_steps:
                break
        m.eval()
        return {"steps": step, "avg_loss": tot / max(step, 1), "train_time_s": round(time.time() - t0, 1)}

    @torch.no_grad()
    def generate(self, aid, text, max_len=64, num_beams=2):
        m = self._model(aid)
        ins = self.tok(text, max_length=512, truncation=True, return_tensors="pt").to(DEVICE)
        out = m.generate(**ins, max_length=max_len, num_beams=num_beams, do_sample=False, early_stopping=True)
        return self.tok.decode(out[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(self, aid, texts, max_len=16, num_beams=1):
        if not texts:
            return []
        m = self._model(aid)
        ins = self.tok(texts, max_length=512, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        out = m.generate(**ins, max_length=max_len, num_beams=num_beams, do_sample=False)
        return [self.tok.decode(o, skip_special_tokens=True) for o in out]


# ── Bedrock-Haiku micro-training event generation ─────────────────────
_REWRITE_P = """Generate {n} JSON pairs for training a query-rewriting model, from this chunk of "{title}".
Each: a vague user "vague_query" and a self-contained retrieval-optimized "explicit_query".
CHUNK: {chunk}
Output ONLY JSON: [{{"vague_query":"...","explicit_query":"..."}}]"""

_RERANK_P = """Generate {n} JSON examples for a relevance-scoring model from this chunk of "{title}".
Each: a "query" and boolean "relevant" (mix true/false: some clearly answerable by the chunk, some not).
CHUNK: {chunk}
Output ONLY JSON: [{{"query":"...","relevant":true}}]"""


def gen_training_events(chunks, title, aux_model=None, n_chunks=None, seed=0):
    aux_model = aux_model or config.AUX_MODEL
    n_chunks = n_chunks or config.TRAIN_CHUNKS
    rng = np.random.default_rng(seed)
    order = sorted(range(len(chunks)), key=lambda i: chunks[i]["page"])
    sample_idx = np.linspace(0, len(order) - 1, num=min(n_chunks, len(order))).astype(int)
    sample = [chunks[order[i]] for i in sample_idx]
    events, cost = [], 0.0
    for ch in sample:
        rw = llm.generate(_REWRITE_P.format(n=2, title=title, chunk=ch["text"][:600]), model=aux_model,
                          temperature=0.6, max_tokens=400)
        cost += rw["cost"]
        for p in (llm.extract_json(rw["text"]) or []):
            if isinstance(p, dict) and p.get("vague_query") and p.get("explicit_query"):
                events.append({"event_type": "rewrite", "input_text": f"rewrite query: {p['vague_query']}",
                               "target_text": p["explicit_query"]})
        rr = llm.generate(_RERANK_P.format(n=3, title=title, chunk=ch["text"][:600]), model=aux_model,
                          temperature=0.6, max_tokens=400)
        cost += rr["cost"]
        for ex in (llm.extract_json(rr["text"]) or []):
            if isinstance(ex, dict) and ex.get("query") is not None and "relevant" in ex:
                events.append({"event_type": "rerank",
                               "input_text": f"rank relevance: query: {ex['query']} document: {ch['text'][:200]}",
                               "target_text": "relevant" if ex["relevant"] else "irrelevant"})
    return events, cost
