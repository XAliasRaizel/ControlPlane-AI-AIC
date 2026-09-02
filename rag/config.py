"""Centralised RAG configuration. Same pattern as backend/shared/config.py:
everything resolved from environment variables with sensible local defaults,
so nothing is hard-coded through the rest of the rag/ package (Section 10).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # repo root

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    _env_file = _ROOT / ".env"
    if _env_file.exists():
        with open(_env_file, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    if _k.strip() not in os.environ:
                        os.environ[_k.strip()] = _v.strip()

DEFAULT_CORPUS_DIR = _ROOT / "rag" / "corpus"
DEFAULT_VECTOR_DB_DIR = _ROOT / "rag_store"  # persisted ChromaDB location


@dataclass(frozen=True)
class RagSettings:
    # --- embeddings ---
    # "tfidf_lsa"            : zero external deps, works offline (default).
    # "sentence_transformers": real pretrained model. Set RAG_EMBEDDING_BACKEND=sentence_transformers
    #                          after `pip install sentence-transformers`.
    embedding_backend: str = os.getenv("RAG_EMBEDDING_BACKEND", "tfidf_lsa")
    # BGE-small gives the best accuracy/speed tradeoff for CPU inference.
    # all-MiniLM-L6-v2 is a lighter alternative.
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
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
    grounding_threshold: float = float(os.getenv("RAG_GROUNDING_THRESHOLD", "0.28"))

    # --- cross-encoder reranker ---
    # Set RAG_RERANK_ENABLED=true to activate after installing sentence-transformers.
    # The reranker fetches `rerank_candidates` chunks first (broad recall),
    # then re-scores them with a cross-encoder and returns the best `top_k`.
    rerank_enabled: bool = os.getenv("RAG_RERANK_ENABLED", "false").lower() == "true"
    rerank_model: str = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_candidates: int = int(os.getenv("RAG_RERANK_CANDIDATES", "20"))

    # --- NLI grounding model ---
    # Set RAG_NLI_ENABLED=true to use DeBERTa-v3-small NLI instead of
    # the lexical LexicalEntailmentChecker. Requires sentence-transformers.
    nli_enabled: bool = os.getenv("RAG_NLI_ENABLED", "false").lower() == "true"
    nli_model: str = os.getenv("RAG_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    nli_entailment_threshold: float = float(os.getenv("RAG_NLI_THRESHOLD", "0.7"))

    # --- BM25 hybrid retrieval ---
    # When bm25_enabled=true, retrieval merges dense vector scores with
    # BM25 sparse scores (RRF fusion). Requires rank_bm25 package.
    bm25_enabled: bool = os.getenv("RAG_BM25_ENABLED", "false").lower() == "true"
    bm25_weight: float = float(os.getenv("RAG_BM25_WEIGHT", "0.3"))

    # --- TF-IDF/LSA stability guard (see LocalTfidfEmbedder.overlap_weight) ---
    # Calibrated against measured examples on the real corpora in this repo:
    # with the expanded corpus (231-line GDPR, 185-line AI Act, etc.),
    # genuine matches score 0.10–0.15 overlap weight while spurious
    # SVD-collapse matches still score 0.0. 0.10 sits with clear margin.
    raw_overlap_guard_threshold: float = float(os.getenv("RAG_RAW_OVERLAP_GUARD_THRESHOLD", "0.10"))

    # --- hot path guard (Section 13) ---
    # Policy RAG is allowed on the hot path only if it can return within this
    # budget; the gateway measures it and falls back to insufficient_evidence
    # rather than blocking the response if it's exceeded.
    hot_path_budget_ms: float = float(os.getenv("RAG_HOT_PATH_BUDGET_MS", "40"))

    # --- LLM generation (Ask ControlPlane) ---
    # When a Groq API key is configured or local Ollama is active, Ask
    # ControlPlane uses the LLM to synthesize answers from retrieved context.
    # Provider options: "auto" (tries Groq if key exists, else Ollama, else extractive),
    #                   "groq" (cloud Groq API),
    #                   "ollama" (local Ollama server).
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "auto")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_max_tokens: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
    groq_temperature: float = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
    generation_enabled: bool = os.getenv("RAG_GENERATION_ENABLED", "true").lower() == "true"

    # --- multi-tenancy (Phase 3) ---
    multi_tenant_enabled: bool = field(
        default_factory=lambda: os.getenv("RAG_MULTI_TENANT_ENABLED", "false").lower() == "true"
    )
    default_tenant_id: str = field(
        default_factory=lambda: os.getenv("RAG_DEFAULT_TENANT_ID", "default")
    )


rag_settings = RagSettings()
