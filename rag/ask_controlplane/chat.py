"""Ask ControlPlane conversational endpoint (spec Section 4/5).

Answer synthesis has two modes, selected automatically at call time:

1. **Generative** (preferred) — retrieved chunks are assembled into a RAG
   context and sent to a Groq-hosted LLM, which produces a coherent,
   grounded answer with light connecting language.  Active when
   ``RAG_GENERATION_ENABLED=true`` (default) **and** ``GROQ_API_KEY`` is
   set **and** the ``groq`` package is installed.

2. **Extractive** (fallback) — the single best-matching chunk's text is
   returned verbatim, exactly as the system worked before Groq integration.
   Activated automatically on *any* failure in the generative path (import
   error, missing key, API timeout, rate limit, empty response, etc.).

``synthesize_answer()`` is still the single, swappable seam — it now
returns a ``(answer_text, generation_mode)`` tuple so the caller can
record which path produced the answer.
"""

from __future__ import annotations

import logging
import re

from rag.ask_controlplane.retrieval import hybrid_retrieve
from rag.config import rag_settings
from rag.schemas import AskControlPlaneAnswer, RetrievedChunk

logger = logging.getLogger("controlplane.rag.ask")

_INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have sufficient evidence in the ControlPlane knowledge base to answer this."
)

_REQUEST_ID_PATTERN = re.compile(r"#?([0-9a-f]{6,}(?:-[0-9a-f]{4,}){0,4}|\d{3,})", re.I)


def _looks_like_request_id_question(question: str) -> str | None:
    """Pulls out something that looks like a request id if the question is
    asking about a specific one ("why was request #4021 blocked")."""
    if not re.search(r"\brequest\b", question, re.I):
        return None
    match = _REQUEST_ID_PATTERN.search(question)
    return match.group(1) if match else None


def _format_citation(chunk: RetrievedChunk) -> str:
    meta = chunk.metadata
    if meta.get("document_type") == "audit_record":
        return f"Audit record {meta.get('request_id', '?')} -> decision {meta.get('decision', '?')}"
    label = meta.get("document") or meta.get("source") or "source"
    section = meta.get("article") or meta.get("policy_id") or ""
    return f"{label}" + (f" -> {section}" if section else "")


def _build_rag_context(chunks: list[RetrievedChunk]) -> str:
    """Assemble retrieved chunks into a single context string for the LLM."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source") or chunk.metadata.get("document") or "unknown"
        parts.append(f"[{i}] (source: {source})\n{chunk.text}")
    return "\n\n".join(parts)


def synthesize_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    """The swappable seam described in the module docstring.

    Returns
    -------
    tuple[str, str]
        (answer_text, generation_mode) where generation_mode is
        ``"groq"`` or ``"extractive"``.
    """
    if not chunks:
        return _INSUFFICIENT_EVIDENCE_MESSAGE, "extractive"

    # --- request-id shortcut (always extractive — exact match, no LLM needed) ---
    requested_id = _looks_like_request_id_question(question)
    if requested_id:
        audit_chunks = [c for c in chunks if c.metadata.get("document_type") == "audit_record"]
        matching = [c for c in audit_chunks if requested_id in c.metadata.get("request_id", "")]
        if matching:
            return matching[0].text, "extractive"
        if not audit_chunks:
            return _INSUFFICIENT_EVIDENCE_MESSAGE, "extractive"

    # --- generative path (Groq) ---
    if rag_settings.generation_enabled and rag_settings.groq_api_key:
        try:
            from rag.ask_controlplane.llm_client import GroqLLMClient

            client = GroqLLMClient()
            context = _build_rag_context(chunks)
            answer = client.generate(context=context, question=question)
            if answer and answer.strip():
                return answer.strip(), "groq"
            # Empty response — fall through to extractive
            logger.warning("Groq returned empty response, falling back to extractive.")
        except Exception as exc:
            logger.warning("Groq generation failed (%s), falling back to extractive.", exc)

    # --- extractive fallback ---
    return chunks[0].text, "extractive"


def ask(question: str, top_k: int = 5, db=None) -> AskControlPlaneAnswer:
    """The Section 4/5 entry point."""
    if not question or not question.strip():
        return AskControlPlaneAnswer(
            answer="Please ask a specific question about a policy, decision, or audit record.",
            citations=[], status="INVALID_REQUEST", confidence=0.0,
        )

    # A question naming a specific request ID is an exact primary-key
    # lookup, not a similarity-search problem -- go straight to the
    # database rather than through semantic/lexical retrieval, which has
    # no reason to be involved when we already have the exact key.
    requested_id = _looks_like_request_id_question(question)
    if requested_id and db is not None:
        audit = db.get_audit(requested_id) or _find_by_id_prefix(db, requested_id)
        if audit:
            from rag.ingestion.audit_loader import audit_record_to_document
            text, metadata = audit_record_to_document(audit)
            chunk = RetrievedChunk(text=text, score=1.0, metadata=metadata)
            return AskControlPlaneAnswer(answer=text, citations=[chunk], status="SUCCESS", confidence=1.0)
        return AskControlPlaneAnswer(
            answer=f"No audit record found matching request id '{requested_id}'.",
            citations=[], status="INSUFFICIENT_EVIDENCE", confidence=0.0,
        )

    try:
        chunks = hybrid_retrieve(question, top_k=top_k)
    except Exception:
        return AskControlPlaneAnswer(
            answer=_INSUFFICIENT_EVIDENCE_MESSAGE, citations=[],
            status="RETRIEVAL_ERROR", confidence=0.0,
        )

    if not chunks:
        return AskControlPlaneAnswer(
            answer=_INSUFFICIENT_EVIDENCE_MESSAGE, citations=[],
            status="INSUFFICIENT_EVIDENCE", confidence=0.0,
        )

    answer_text, generation_mode = synthesize_answer(question, chunks)
    confidence = round(sum(c.score for c in chunks[:3]) / min(3, len(chunks)), 3)

    return AskControlPlaneAnswer(
        answer=answer_text, citations=chunks, status="SUCCESS",
        confidence=confidence, generation_mode=generation_mode,
    )


def _find_by_id_prefix(db, prefix: str):
    """request_id is a full UUID; questions typically only quote a short
    prefix of it ("#2d015591"). Scan recent audits for a prefix match
    rather than requiring the exact full UUID."""
    for audit in db.recent_audits(limit=500):
        if audit["request_id"].startswith(prefix):
            return audit
    return None
