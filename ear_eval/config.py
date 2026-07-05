"""Central configuration for the harmonized EAR evaluation harness (Bedrock)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

ENV_PATH = Path(os.environ.get("EAR_ENV_PATH", "/Users/mac001/Documents/WMP-CRIS/.env"))


def load_env():
    """(Re)load AWS creds + keys from the project .env into os.environ."""
    if load_dotenv and ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)


load_env()

AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")

# ── Bedrock models ────────────────────────────────────────────────────
SONNET = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"   # benchmarked answer generation
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"     # auxiliary work + LLM judge
TITAN = "amazon.titan-embed-text-v2:0"                    # embeddings

# Bedrock on-demand pricing, USD per 1M tokens
PRICING = {
    SONNET: {"input": 3.00, "output": 15.00},
    HAIKU:  {"input": 1.00, "output": 5.00},
    TITAN:  {"input": 0.02, "output": 0.0},
}

# Default model roles (overridable for the Tier-3 generator swap)
GEN_MODEL = SONNET      # all benchmarked RAG generation
AUX_MODEL = HAIKU       # question gen/verify, LoRA training events
JUDGE_MODEL = HAIKU     # harmonized LLM judge
JUDGE_MODEL_B = SONNET  # second judge for agreement / human-eval cross-check

# ── Retrieval / chunking ──────────────────────────────────────────────
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 6               # simple-RAG retrieval depth
RERANK_FETCH = 20       # deep retrieval for reranked baselines + EAR
RERANK_TOP_N = 5        # keep after cross-encoder rerank
TOKEN_BUDGET = 2200     # packed-context budget (tokens, ~4 chars/token)
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Controller / EAR ──────────────────────────────────────────────────
COVERAGE_THRESHOLD = 0.7
MAX_EAR_ITERATIONS = 2
MAX_AGENTIC_ITERATIONS = 3

# ── EAR LoRA (Phase 3B) ───────────────────────────────────────────────
SLM_MODEL_NAME = "google/flan-t5-small"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q", "v"]
TRAIN_CHUNKS = 30       # sampled chunks used to synthesize LoRA training events

# ── Judge ─────────────────────────────────────────────────────────────
JUDGE_RUNS = 3          # repeat judge at temp 0 and report mean ± std

# ── Paths ─────────────────────────────────────────────────────────────
RESULTS_DIR = Path(os.environ.get(
    "EAR_RESULTS_DIR", "/Users/mac001/Documents/WMP-CRIS/evaluation/harmonized"))
WORK_DIR = Path(os.environ.get(
    "EAR_WORK_DIR",
    "/private/tmp/claude-504/-Users-mac001-Documents-WMP-CRIS/a6304f15-8a8f-4b0c-bc1b-a2eff3fc3622/scratchpad/ear_work"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
