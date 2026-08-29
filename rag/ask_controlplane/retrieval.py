"""Ask ControlPlane hybrid retrieval (spec Section 5):

    Reviewer Question -> Query Understanding -> Hybrid Retrieval
                                    (Audit RAG + Policy RAG)
                              -> Retrieved Context

Combines the policy_evidence collection (regulatory + policy prose,
already built for Policy RAG) with a live-rebuilt audit collection
(strict allow-list -- see rag/ingestion/audit_loader.py) into one ranked
context list.
"""

from __future__ import annotations

from pathlib import Path

from rag.config import rag_settings
from rag.embeddings import LocalTfidfEmbedder, get_embedder
from rag.ingestion.audit_loader import load_audit_corpus
from rag.retriever import Retriever
from rag.schemas import RetrievedChunk
from rag.vector_store import VectorStore

_POLICY_EMBEDDER_PATH = Path(rag_settings.vector_store_dir) / "policy_embedder.pkl"
_AUDIT_EMBEDDER_PATH = Path(rag_settings.vector_store_dir) / "audit_embedder.pkl"

_policy_retriever: Retriever | None = None
_audit_retriever: Retriever | None = None


def _get_policy_retriever() -> Retriever:
    global _policy_retriever
    if _policy_retriever is None:
        if not _POLICY_EMBEDDER_PATH.exists():
            try:
                from rag.ingestion.ingest import ingest
                ingest()
            except Exception:
                pass
        embedder = get_embedder()
        if isinstance(embedder, LocalTfidfEmbedder) and _POLICY_EMBEDDER_PATH.exists():
            embedder.load(_POLICY_EMBEDDER_PATH)
        _policy_retriever = Retriever("policy_evidence", embedder=embedder)
    return _policy_retriever


def rebuild_audit_index(db, limit: int = 500) -> int:
    """Rebuilds the audit collection from live data (spec Section 4/8).
    Unlike the static policy/regulatory corpus, this is generated from the
    database, not files, and grows as new governance decisions happen --
    called from the /v1/ask-controlplane/reindex endpoint (or a scheduled
    job in a real deployment) rather than once at boot.
    """
    audits = db.recent_audits(limit=limit)
    chunks = load_audit_corpus(audits)
    if not chunks:
        return 0

    embedder = get_embedder()
    embedder.fit([c.text for c in chunks])
    embedder.save(_AUDIT_EMBEDDER_PATH)
    vectors = embedder.embed([c.text for c in chunks])

    store = VectorStore("audit_log")
    store.reset()
    store.upsert(
        ids=[c.chunk_id for c in chunks],
        texts=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[c.metadata for c in chunks],
    )

    global _audit_retriever
    _audit_retriever = Retriever("audit_log", embedder=embedder)
    return len(chunks)


def _get_audit_retriever() -> Retriever | None:
    global _audit_retriever
    if _audit_retriever is None and _AUDIT_EMBEDDER_PATH.exists():
        embedder = get_embedder()
        if isinstance(embedder, LocalTfidfEmbedder):
            embedder.load(_AUDIT_EMBEDDER_PATH)
        _audit_retriever = Retriever("audit_log", embedder=embedder)
    return _audit_retriever


def hybrid_retrieve(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Section 5's fan-out: query both collections, merge by score. Either
    collection failing (e.g. audit index not built yet) degrades to
    whichever one succeeded, rather than failing the whole question.
    """
    results: list[RetrievedChunk] = []

    policy_result = _get_policy_retriever().retrieve(question, top_k=top_k)
    if policy_result.status == "SUCCESS":
        results.extend(policy_result.chunks)

    audit_retriever = _get_audit_retriever()
    if audit_retriever is not None:
        audit_result = audit_retriever.retrieve(question, top_k=top_k)
        if audit_result.status == "SUCCESS":
            results.extend(audit_result.chunks)

    results.sort(key=lambda c: c.score, reverse=True)
    return results[:top_k]
