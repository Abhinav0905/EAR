"""Harmonized EAR-RCAA evaluation harness (Bedrock: Sonnet generation, Haiku auxiliary/judge).

HF offline by default (flan-t5 + cross-encoder are cached; the corporate proxy blocks HF SSL).
Set HF_HUB_OFFLINE=0 + SSL_CERT_FILE=<combined CA> in the environment for steps that must
download (e.g. the QASPER dataset). Must run before any transformers/huggingface_hub import.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
