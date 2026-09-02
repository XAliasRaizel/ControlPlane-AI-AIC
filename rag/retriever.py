"""The retriever abstraction (spec Section 9): one clean call surface that
every one of the three RAG systems (Policy RAG, Grounding RAG, Ask
ControlPlane) uses, so retrieval behavior — thresholds, error handling,
insufficient-evidence logic — lives in exactly one place.

Upgrades in this version:
  1. Cross-Encoder Reranker  — when RAG_RERANK_ENABLED=true, fetches
     `rerank_candidates` chunks first, then re-scores every (query, chunk)
     pair with a cross-encoder and returns the best `top_k`. Huge precision
     boost with no change to the caller API.
  2. BM25 Hybrid Retrieval   — when RAG_BM25_ENABLED=true, merges dense
     cosine scores with BM25 sparse keyword scores via Reciprocal Rank
     Fusion (RRF). Helps exact-term queries (policy IDs, article numbers).
  3. Domain Metadata Filtering — callers pass filters={"domain": "hr"} and
     it flows to both ChromaDB and the SimpleVectorStore WHERE clause.
  4. Lazy-loaded singletons   — cross-encoder built once on first use.

Both features degrade gracefully: if sentence-transformers or rank_bm25
are not installed, code falls back to pure vector retrieval + a one-time
warning log. Nothing crashes.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from rag.config import rag_settings
from rag.embeddings import BaseEmbedder, LocalTfidfEmbedder, get_embedder
from rag.schemas import RetrievalResult, RetrievedChunk
from rag.vector_store import VectorStore

logger = logging.getLogger("controlplane.rag")

# ---------------------------------------------------------------------------
# Lazy-loaded singleton: Cross-Encoder reranker
# ---------------------------------------------------------------------------
_reranker = None
_reranker_loaded: bool = False


def _get_reranker():
    """Load cross-encoder once; return None and warn on first failure."""
    global _reranker, _reranker_loaded
    if _reranker_loaded:
        return _reranker
    _reranker_loaded = True
    if not rag_settings.rerank_enabled:
        return None
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        _reranker = CrossEncoder(rag_settings.rerank_model, max_length=512)
        logger.info("RAG reranker loaded: %s", rag_settings.rerank_model)
    except Exception as exc:
        logger.warning(
            "Cross-encoder reranker unavailable (%s). "
            "Run: pip install sentence-transformers  then set RAG_RERANK_ENABLED=true. "
            "Falling back to vector-only retrieval.",
            exc,
        )
        _reranker = None
    return _reranker


def _rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-score chunks with the cross-encoder, return top_k by new score."""
    reranker = _get_reranker()
    if reranker is None or not chunks:
        return chunks[:top_k]
    try:
        pairs = [[query, c.text] for c in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: float(x[0]), reverse=True)
        result = []
        for raw_score, chunk in ranked[:top_k]:
            # Normalise cross-encoder logit to [0,1] via sigmoid
            normalised = 1.0 / (1.0 + math.exp(-float(raw_score)))
            result.append(RetrievedChunk(
                text=chunk.text,
                score=round(normalised, 4),
                metadata=chunk.metadata,
            ))
        return result
    except Exception as exc:
        logger.warning("Reranker prediction failed (%s), using original order.", exc)
        return chunks[:top_k]


# ---------------------------------------------------------------------------
# BM25 index per collection
# ---------------------------------------------------------------------------
_bm25_indexes: dict[str, Any] = {}
_bm25_corpus: dict[str, list[str]] = {}


def _get_bm25(collection_name: str, texts: list[str]):
    """Build or return cached BM25 index. Returns None if unavailable."""
    if not rag_settings.bm25_enabled:
        return None
    cached_size = len(_bm25_corpus.get(collection_name, []))
    if cached_size == len(texts) and collection_name in _bm25_indexes:
        return _bm25_indexes[collection_name]
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
        tokenized = [t.lower().split() for t in texts]
        _bm25_indexes[collection_name] = BM25Okapi(tokenized)
        _bm25_corpus[collection_name] = list(texts)
        return _bm25_indexes[collection_name]
    except ImportError:
        logger.warning("rank_bm25 not installed — BM25 hybrid disabled. Run: pip install rank_bm25")
        return None
    except Exception as exc:
        logger.warning("BM25 index build failed (%s), using vector-only.", exc)
        return None


def _bm25_scores(bm25, query: str, texts: list[str]) -> list[float]:
    """Normalised BM25 scores for query against texts."""
    try:
        import numpy as np
        raw = bm25.get_scores(query.lower().split())
        max_score = float(np.max(raw)) if len(raw) > 0 else 1.0
        if max_score == 0:
            max_score = 1.0
        return [float(s / max_score) for s in raw]
    except Exception:
        return [0.0] * len(texts)


def _rrf_fusion(
    chunks: list[RetrievedChunk],
    b_scores: list[float],
    bm25_weight: float = 0.3,
) -> list[RetrievedChunk]:
    """Blend dense vector score with BM25 sparse score (Reciprocal Rank Fusion)."""
    if not b_scores or len(b_scores) != len(chunks):
        return chunks
    fused = []
    for chunk, b_score in zip(chunks, b_scores):
        fused_score = (1.0 - bm25_weight) * chunk.score + bm25_weight * b_score
        fused.append(RetrievedChunk(
            text=chunk.text,
            score=round(fused_score, 4),
            metadata=chunk.metadata,
        ))
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused


# ---------------------------------------------------------------------------
# Main Retriever
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, collection_name: str, embedder: BaseEmbedder | None = None, tenant_id: str = "default"):
        self.collection_name = collection_name
        self.tenant_id = tenant_id
        self.embedder = embedder or get_embedder()
        self.store = VectorStore(collection_name, tenant_id=tenant_id)

    def retrieve(
        self,
        query: str,
        top_k: int = rag_settings.top_k,
        filters: dict | None = None,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        """Section 9's exact call surface — always returns a RetrievalResult,
        never raises. Supports metadata filtering, BM25 hybrid fusion, and
        cross-encoder reranking transparently.
        """
        if not query or not query.strip():
            return RetrievalResult(status="INVALID_REQUEST", query=query, error="Empty query")

        if not self.embedder.is_fitted:
            return RetrievalResult(
                status="RETRIEVAL_ERROR", query=query,
                error=f"Embedder for '{self.collection_name}' is not fitted — run ingestion first.",
            )

        # Fetch more candidates when reranking (broad recall → precise rerank)
        fetch_k = rag_settings.rerank_candidates if rag_settings.rerank_enabled else top_k

        try:
            start = time.perf_counter()
            query_vec = self.embedder.embed([query])[0]
            raw = self.store.query(query_vec, top_k=fetch_k, where=filters)
            elapsed_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            logger.warning("Retrieval failed for '%s': %s", self.collection_name, exc)
            return RetrievalResult(status="RETRIEVAL_ERROR", query=query, error=str(exc))

        chunks = [
            RetrievedChunk(text=r["text"], score=r["score"], metadata=r["metadata"])
            for r in raw
            if r["score"] >= min_score
        ]

        # TF-IDF/LSA stability guard (see LocalTfidfEmbedder.overlap_weight docstring)
        if chunks and isinstance(self.embedder, LocalTfidfEmbedder):
            filtered = []
            for c in chunks:
                weight = self.embedder.overlap_weight(query, c.text)
                if weight >= rag_settings.raw_overlap_guard_threshold:
                    filtered.append(c)
                else:
                    logger.info(
                        "Retrieval for '%s': dropping chunk with similarity %.3f "
                        "but overlap weight only %.3f.",
                        self.collection_name, c.score, weight,
                    )
            chunks = filtered

        if not chunks:
            return RetrievalResult(status="INSUFFICIENT_EVIDENCE", query=query, chunks=[])

        # BM25 hybrid fusion (RRF)
        if rag_settings.bm25_enabled and chunks:
            all_texts = [c.text for c in chunks]
            bm25_key = f"{self.tenant_id}__{self.collection_name}" if self.tenant_id != "default" else self.collection_name
            bm25 = _get_bm25(bm25_key, all_texts)
            if bm25 is not None:
                b_scores = _bm25_scores(bm25, query, all_texts)
                chunks = _rrf_fusion(chunks, b_scores, rag_settings.bm25_weight)

        # Cross-encoder reranking
        if rag_settings.rerank_enabled:
            chunks = _rerank(query, chunks, top_k)
            # Filter out chunks that the cross-encoder scored as virtually zero (< 0.01)
            chunks = [c for c in chunks if c.score >= 0.01]
        else:
            chunks = chunks[:top_k]

        if not chunks:
            return RetrievalResult(status="INSUFFICIENT_EVIDENCE", query=query, chunks=[])

        if elapsed_ms > rag_settings.hot_path_budget_ms:
            logger.warning(
                "Retrieval for '%s' took %.1fms, over the %.0fms hot-path budget",
                self.collection_name, elapsed_ms, rag_settings.hot_path_budget_ms,
            )

        return RetrievalResult(status="SUCCESS", query=query, chunks=chunks)


