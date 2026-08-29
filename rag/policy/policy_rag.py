"""Policy RAG (spec Section 2): retrieval-based policy evidence, layered on
top of -- never replacing -- the deterministic Policy Engine.

    Hard Rules  +  Policy RAG Evidence  +  Context
                        |
                 Policy / Decision Engine
                        |
                ALLOW / BLOCK / REVIEW

Deterministic rules (backend/policy/engine.py) make the decision. This
module runs AFTER that decision, and only explains/documents it with
retrieved evidence -- it never changes what was decided. That's a
deliberate architectural choice, not a limitation: the spec is explicit
that RAG should provide "policy knowledge and evidence, while deterministic
rules remain responsible for critical enforcement."
"""

from __future__ import annotations

import time
from pathlib import Path

from rag.config import rag_settings
from rag.embeddings import LocalTfidfEmbedder, get_embedder
from rag.retriever import Retriever
from rag.schemas import PolicyEvidence

_POLICY_EMBEDDER_PATH = Path(rag_settings.vector_store_dir) / "policy_embedder.pkl"

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    """Lazy singleton: load the persisted, already-fitted policy index once
    per process, not on every call (Section 7: don't rebuild on startup)."""
    global _retriever
    if _retriever is None:
        if not _POLICY_EMBEDDER_PATH.exists():
            try:
                from rag.ingestion.ingest import ingest
                ingest()
            except Exception:
                pass
        embedder = get_embedder()
        if isinstance(embedder, LocalTfidfEmbedder) and _POLICY_EMBEDDER_PATH.exists():
            embedder.load(_POLICY_EMBEDDER_PATH)
        _retriever = Retriever("policy_evidence", embedder=embedder)
    return _retriever


def build_policy_query(
    *,
    application_id: str,
    department: str | None,
    user_role: str,
    action: str,
    data_classification: str | None,
    matched_rule_description: str | None = None,
) -> str:
    """Constructs a semantic query from decision context (spec's own
    example: "employee accessing salary data without authorization, EU
    jurisdiction"). Deliberately short and keyword-dense -- this is TF-IDF
    territory, not a place for a long natural sentence.
    """
    parts = [user_role or "user", "in", application_id]
    if department:
        parts.append(f"({department})")
    if matched_rule_description:
        parts.append(matched_rule_description)
    else:
        parts.append(f"resulting in {action}")
    if data_classification:
        parts.append(f"data classification {data_classification}")
    return " ".join(parts)


def get_policy_evidence(
    *,
    application_id: str,
    department: str | None,
    user_role: str,
    action: str,
    data_classification: str | None,
    matched_rule_description: str | None = None,
    top_k: int = 3,
) -> PolicyEvidence:
    """The Section 2 entry point. Always returns a PolicyEvidence with an
    explicit status -- never raises, never silently returns nothing.
    Bounded for hot-path use (Section 13): if retrieval exceeds the
    configured budget, returns insufficient_evidence rather than blocking
    the response, exactly like a low-confidence result would.
    """
    query = build_policy_query(
        application_id=application_id, department=department, user_role=user_role,
        action=action, data_classification=data_classification,
        matched_rule_description=matched_rule_description,
    )

    start = time.perf_counter()
    result = _get_retriever().retrieve(
        query, top_k=top_k, min_score=rag_settings.policy_retrieval_threshold
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if elapsed_ms > rag_settings.hot_path_budget_ms:
        return PolicyEvidence(status="insufficient_evidence", query=query, citations=[],
                               summary="Policy retrieval exceeded the hot-path time budget.")

    if result.status != "SUCCESS":
        return PolicyEvidence(status="insufficient_evidence", query=query, citations=[])

    # Extractive summary -- a formatted list of what was found, not a
    # generated claim (Section 13: no generation step in the critical path).
    summary = "; ".join(
        f"{c.metadata.get('document', c.metadata.get('source', 'evidence'))}" for c in result.chunks[:3]
    )
    return PolicyEvidence(status="SUCCESS", query=query, citations=result.chunks, summary=summary)
