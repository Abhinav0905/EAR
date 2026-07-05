"""Bedrock LLM (Sonnet/Haiku) + Titan embeddings, with creds-reload + backoff.

Designed for long, resumable runs on short-lived STS tokens: on ExpiredToken the
client is rebuilt from a fresh read of .env, so refreshing creds mid-run recovers.
"""
import json
import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError

from . import config

_client = None

_RETRYABLE = {
    "ThrottlingException", "TooManyRequestsException", "ModelTimeoutException",
    "ServiceUnavailableException", "InternalServerException", "ModelErrorException",
}
_CRED = {"ExpiredTokenException", "UnrecognizedClientException", "ExpiredToken",
         "InvalidSignatureException", "AccessDeniedException"}


def _new_client():
    config.load_env()
    return boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", config.AWS_REGION),
        config=Config(retries={"max_attempts": 2, "mode": "adaptive"},
                      read_timeout=180, connect_timeout=20),
    )


def client():
    global _client
    if _client is None:
        _client = _new_client()
    return _client


def _reset_client():
    global _client
    _client = _new_client()
    return _client


def cost(model, in_tok, out_tok):
    p = config.PRICING.get(model, {"input": 0.0, "output": 0.0})
    return in_tok / 1e6 * p["input"] + out_tok / 1e6 * p["output"]


def extract_json(text):
    """Parse JSON from an LLM response that may be markdown-fenced or wrapped in prose."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None


def extract_dict(text, default=None):
    """Like extract_json but guarantees a dict (LLMs sometimes return a bare list/scalar)."""
    d = extract_json(text)
    return d if isinstance(d, dict) else (default if default is not None else {})


OPENAI_PRICING = {"gpt-4o-mini": {"input": 0.15, "output": 0.60}}  # USD / 1M tokens
_oai_client = None


def _is_openai(model):
    return bool(model) and str(model).startswith("gpt")


def _oai():
    global _oai_client
    if _oai_client is None:
        import httpx
        from openai import OpenAI
        config.load_env()
        key = os.environ.get("OPENAI_KEYS") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("no OPENAI_KEYS / OPENAI_API_KEY in env")
        _oai_client = OpenAI(api_key=key, http_client=httpx.Client(timeout=90))
    return _oai_client


def _generate_openai(prompt, model, system, temperature, max_tokens, tries=8):
    """OpenAI chat-completion generator; returns the same dict shape as generate()."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    if isinstance(prompt, list):
        msgs += prompt
    else:
        msgs.append({"role": "user", "content": prompt})
    last = None
    for attempt in range(tries):
        try:
            t0 = time.time()
            r = _oai().chat.completions.create(model=model, messages=msgs,
                                               temperature=temperature, max_tokens=max_tokens)
            u = r.usage
            it, ot = u.prompt_tokens, u.completion_tokens
            p = OPENAI_PRICING.get(model, {"input": 0.0, "output": 0.0})
            return {"text": r.choices[0].message.content or "", "input_tokens": it, "output_tokens": ot,
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                    "cost": it / 1e6 * p["input"] + ot / 1e6 * p["output"], "model": model}
        except Exception as e:
            last = e
            m = str(e).lower()
            if any(k in m for k in ["429", "rate", "timeout", "connection", "overloaded", "503", "502"]):
                time.sleep(min(30, 2 ** attempt + random.random())); continue
            if attempt < 2:
                time.sleep(2); continue
            raise
    raise last


def generate(prompt, model=None, system=None, temperature=0.0, max_tokens=1024, tries=8):
    """Invoke a generation model. `prompt` is a str or a messages list. Routes OpenAI
    models (gpt-*) to the OpenAI API and everything else to Bedrock Anthropic.

    Returns {text, input_tokens, output_tokens, latency_ms, cost, model}.
    """
    model = model or config.GEN_MODEL
    if _is_openai(model):
        return _generate_openai(prompt, model, system, temperature, max_tokens, tries=tries)
    msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": max_tokens,
            "temperature": temperature, "messages": msgs}
    if system:
        body["system"] = system
    payload = json.dumps(body)
    last = None
    cred_tries = 0
    for attempt in range(tries):
        try:
            t0 = time.time()
            r = client().invoke_model(modelId=model, body=payload)
            o = json.loads(r["body"].read())
            blocks = o.get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type", "text") == "text")
            u = o.get("usage", {})
            it, ot = u.get("input_tokens", 0), u.get("output_tokens", 0)
            return {"text": text, "input_tokens": it, "output_tokens": ot,
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                    "cost": cost(model, it, ot), "model": model}
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            last = e
            if code in _CRED:
                # token likely expired: reload .env once, retry briefly, then fail fast
                cred_tries += 1
                _reset_client()
                if cred_tries > 2:
                    raise
                time.sleep(3)
                continue
            if code in _RETRYABLE:
                time.sleep(min(45, 2 ** attempt + random.random()))
                continue
            raise
        except (BotoCoreError, Exception) as e:
            last = e
            time.sleep(min(20, 2 ** attempt))
            continue
    raise last


class TitanEmbeddings:
    """langchain-compatible embeddings using Bedrock Titan v2."""

    def __init__(self, model=None, max_workers=8):
        self.model = model or config.TITAN
        self.max_workers = max_workers

    def _one(self, text):
        body = json.dumps({"inputText": (text or " ")[:8000]})
        for attempt in range(8):
            try:
                r = client().invoke_model(modelId=self.model, body=body)
                return json.loads(r["body"].read())["embedding"]
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in _CRED:
                    _reset_client(); time.sleep(min(30, 5 * (attempt + 1))); continue
                if code in _RETRYABLE:
                    time.sleep(min(45, 2 ** attempt + random.random())); continue
                raise
            except Exception:
                time.sleep(min(20, 2 ** attempt)); continue
        raise RuntimeError("Titan embedding failed after retries")

    def embed_documents(self, texts):
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(self._one, texts))

    def embed_query(self, text):
        return self._one(text)
