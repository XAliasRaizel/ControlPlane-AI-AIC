"""Ask ControlPlane conversational endpoint (spec Section 4/5).

Answer synthesis here is EXTRACTIVE, not generative -- composed directly
from retrieved chunk text plus light connecting language, not produced by
an LLM call. This mirrors the existing llm_simulator.py pattern already
used elsewhere in this codebase (a real generative LLM needs an API key
this sandbox doesn't have configured) and is honest about what it is: no
invented facts are possible because nothing is generated, only assembled
from what was actually retrieved. `synthesize_answer()` is written as a
single, swappable seam -- replace its body with a real LLM call
(retrieved chunks as context, standard RAG prompting) and nothing else in
the pipeline needs to change.
"""

from __future__ import annotations

import re

from rag.ask_controlplane.retrieval import hybrid_retrieve
from rag.schemas import AskControlPlaneAnswer, RetrievedChunk

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


def synthesize_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """The swappable seam described in the module docstring."""
    if not chunks:
        return _INSUFFICIENT_EVIDENCE_MESSAGE

    requested_id = _looks_like_request_id_question(question)
    if requested_id:
        audit_chunks = [c for c in chunks if c.metadata.get("document_type") == "audit_record"]
        matching = [c for c in audit_chunks if requested_id in c.metadata.get("request_id", "")]
        if matching:
            return matching[0].text
        if not audit_chunks:
            return _INSUFFICIENT_EVIDENCE_MESSAGE

    # General case: lead with the single best-matching chunk's text,
    # verbatim (it's already well-formed prose from the ingestion
    # pipeline), rather than trying to further compress or paraphrase it
    # without a generation step.
    return chunks[0].text


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

    answer_text = synthesize_answer(question, chunks)
    confidence = round(sum(c.score for c in chunks[:3]) / min(3, len(chunks)), 3)

    return AskControlPlaneAnswer(
        answer=answer_text, citations=chunks, status="SUCCESS", confidence=confidence,
    )


def _find_by_id_prefix(db, prefix: str):
    """request_id is a full UUID; questions typically only quote a short
    prefix of it ("#2d015591"). Scan recent audits for a prefix match
    rather than requiring the exact full UUID."""
    for audit in db.recent_audits(limit=500):
        if audit["request_id"].startswith(prefix):
            return audit
    return None
