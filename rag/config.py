"""Centralised RAG configuration. Same pattern as backend/shared/config.py:
everything resolved from environment variables with sensible local defaults,
so nothing is hard-coded through the rest of the rag/ package (Section 10).
"""

import os
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # repo root

DEFAULT_CORPUS_DIR = _ROOT / "rag" / "corpus"
DEFAULT_VECTOR_DB_DIR = _ROOT / "rag_store"  # persisted ChromaDB location


@dataclass(frozen=True)
class RagSettings:
    # --- embeddings ---
    # "tfidf_lsa" works with zero external dependencies or model downloads
    # (see rag/embeddings.py for why this is the default here). Set to
    # "sentence_transformers" in an environment with normal internet access
    # to use a real pretrained model instead -- no other code changes needed.
    embedding_backend: str = os.getenv("RAG_EMBEDDING_BACKEND", "tfidf_lsa")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dims: int = int(os.getenv("RAG_EMBEDDING_DIMS", "128"))

    # --- vector store ---
    vector_store_dir: str = os.getenv("RAG_VECTOR_STORE_DIR", str(DEFAULT_VECTOR_DB_DIR))
    corpus_dir: str = os.getenv("RAG_CORPUS_DIR", str(DEFAULT_CORPUS_DIR))

    # --- chunking ---
    chunk_size_chars: int = int(os.getenv("RAG_CHUNK_SIZE_CHARS", "800"))
    chunk_overlap_chars: int = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "120"))

    # --- retrieval ---
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    policy_retrieval_threshold: float = float(os.getenv("RAG_POLICY_THRESHOLD", "0.20"))
    grounding_threshold: float = float(os.getenv("RAG_GROUNDING_THRESHOLD", "0.35"))
    rerank_enabled: bool = os.getenv("RAG_RERANK_ENABLED", "false").lower() == "true"

    # --- TF-IDF/LSA stability guard (see LocalTfidfEmbedder.overlap_weight) ---
    # Calibrated against measured examples on the real corpora in this repo:
    # a genuine match scored 0.517 unnormalized overlap weight; a spurious
    # SVD-collapse match scored 0.0. 0.15 sits with wide margin on both sides.
    raw_overlap_guard_threshold: float = float(os.getenv("RAG_RAW_OVERLAP_GUARD_THRESHOLD", "0.15"))

    # --- hot path guard (Section 13) ---
    # Policy RAG is allowed on the hot path only if it can return within this
    # budget; the gateway measures it and falls back to insufficient_evidence
    # rather than blocking the response if it's exceeded.
    hot_path_budget_ms: float = float(os.getenv("RAG_HOT_PATH_BUDGET_MS", "40"))


rag_settings = RagSettings()
