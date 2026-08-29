"""The retriever abstraction (spec Section 9): one clean call surface that
every one of the three RAG systems (Policy RAG, Grounding RAG, Ask
ControlPlane) uses, so retrieval behavior -- thresholds, error handling,
insufficient-evidence logic -- lives in exactly one place.
"""

from __future__ import annotations

import logging
import time

from rag.config import rag_settings
from rag.embeddings import BaseEmbedder, LocalTfidfEmbedder, get_embedder
from rag.schemas import RetrievalResult, RetrievedChunk
from rag.vector_store import VectorStore

logger = logging.getLogger("controlplane.rag")


class Retriever:
    def __init__(self, collection_name: str, embedder: BaseEmbedder | None = None):
        self.collection_name = collection_name
        self.embedder = embedder or get_embedder()
        self.store = VectorStore(collection_name)

    def retrieve(
        self,
        query: str,
        top_k: int = rag_settings.top_k,
        filters: dict | None = None,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        """Section 9's exact call surface. Always returns a RetrievalResult
        with an explicit status -- never raises out to the caller, never
        silently returns an empty list with no explanation (Section 15).
        """
        if not query or not query.strip():
            return RetrievalResult(status="INVALID_REQUEST", query=query, error="Empty query")

        if not self.embedder.is_fitted:
            return RetrievalResult(
                status="RETRIEVAL_ERROR", query=query,
                error=f"Embedder for '{self.collection_name}' is not fitted -- run ingestion first.",
            )

        try:
            start = time.perf_counter()
            query_vec = self.embedder.embed([query])[0]
            raw = self.store.query(query_vec, top_k=top_k, where=filters)
            elapsed_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            logger.warning("Retrieval failed for '%s': %s", self.collection_name, exc)
            return RetrievalResult(status="RETRIEVAL_ERROR", query=query, error=str(exc))

        chunks = [
            RetrievedChunk(text=r["text"], score=r["score"], metadata=r["metadata"])
            for r in raw
            if r["score"] >= min_score
        ]

        # Stability guard (see LocalTfidfEmbedder.overlap_weight's
        # docstring for the two measures that were tried and failed first,
        # and why this one held up): a spurious SVD-space score can even
        # outrank a genuinely better match (measured directly: a 6-chunk
        # corpus ranked a weakly-related chunk at 0.999 ahead of the
        # actually-best match at 0.996). Filtering only chunks[0] would
        # have discarded the whole result set on a bad top rank instead of
        # falling through to the good match right behind it -- so each
        # chunk is checked and filtered individually, not gated as a block.
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

        if elapsed_ms > rag_settings.hot_path_budget_ms:
            logger.warning(
                "Retrieval for '%s' took %.1fms, over the %.0fms hot-path budget",
                self.collection_name, elapsed_ms, rag_settings.hot_path_budget_ms,
            )

        return RetrievalResult(status="SUCCESS", query=query, chunks=chunks)
