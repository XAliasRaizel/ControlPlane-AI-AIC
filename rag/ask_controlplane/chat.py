"""Ask ControlPlane conversational endpoint (spec Section 4/5).

Answer synthesis has two modes, selected automatically at call time:

1. **Generative** (preferred) -- retrieved chunks are assembled into an
   injection-safe evidence block (build_evidence_block) and sent to a
   shared LLMClient, which produces a grounded answer and verifies that
   every [N] citation refers to real retrieved evidence (verify_citations).
   Active when RAG_GENERATION_ENABLED=true (default) AND GROQ_API_KEY is
   set AND the groq package is installed.

2. **Extractive** (fallback) -- default_extractive_fallback() lists
   retrieved evidence directly. Never confused with LLM prose because it
   labels itself explicitly.

NOTE: This path is NOT part of the hot-path detector pipeline. It runs on
a separate, slower path with its own latency budget and is never inserted
into or blocking the sub-50ms governance decisions.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

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


def _get_llm_client():
    """Lazily construct the shared LLMClient using rag_settings."""
    from backend.app.llm.client import LLMClient
    return LLMClient(
        api_key_getter=lambda: rag_settings.groq_api_key or None,
        model=rag_settings.groq_model,
        max_completion_tokens=rag_settings.groq_max_tokens,
    )


def synthesize_answer(
    question: str,
    chunks: list[RetrievedChunk],
) -> tuple[str, str, Optional[dict]]:
    """The swappable seam described in the module docstring.

    Returns
    -------
    tuple[str, str, Optional[dict]]
        (answer_text, generation_mode, citation_check)
        generation_mode: "llm" | "extractive"
        citation_check: dict from verify_citations(), or None for extractive
    """
    if not chunks:
        return _INSUFFICIENT_EVIDENCE_MESSAGE, "extractive", None

    # --- request-id shortcut (always extractive -- exact match, no LLM needed) ---
    requested_id = _looks_like_request_id_question(question)
    if requested_id:
        audit_chunks = [c for c in chunks if c.metadata.get("document_type") == "audit_record"]
        matching = [c for c in audit_chunks if requested_id in c.metadata.get("request_id", "")]
        if matching:
            return matching[0].text, "extractive", None
        if not audit_chunks:
            return _INSUFFICIENT_EVIDENCE_MESSAGE, "extractive", None

    # --- generative path via shared LLMClient ---
    if rag_settings.generation_enabled and rag_settings.groq_api_key:
        try:
            from backend.app.llm.prompts import build_chatbot_system_prompt

            client = _get_llm_client()
            context_texts = [c.text for c in chunks]
            response = client.generate(
                system_prompt=build_chatbot_system_prompt(),
                user_prompt=question,
                context=context_texts,
            )

            if response.generation_mode == "llm" and response.text.strip():
                if response.citation_check and not response.citation_check["ok"]:
                    logger.warning(
                        "Ask ControlPlane citation check failed: invalid citations %s",
                        response.citation_check["invalid_citations"],
                    )
                return response.text.strip(), "llm", response.citation_check

            logger.warning(
                "LLMClient returned extractive mode (error=%s), propagating fallback.",
                response.error,
            )
            return response.text, "extractive", None

        except Exception as exc:
            logger.warning("LLM layer failed (%s), falling back to extractive.", exc)

    # --- extractive fallback ---
    from backend.app.llm.client import default_extractive_fallback
    context_texts = [c.text for c in chunks]
    return default_extractive_fallback(question, context_texts), "extractive", None


def ask(question: str, top_k: int = 5, db=None) -> AskControlPlaneAnswer:
    """The Section 4/5 entry point.

    NOTE: Not part of the hot-path. Runs on its own latency budget.
    """
    if not question or not question.strip():
        return AskControlPlaneAnswer(
            answer="Please ask a specific question about a policy, decision, or audit record.",
            citations=[], status="INVALID_REQUEST", confidence=0.0,
        )

    requested_id = _looks_like_request_id_question(question)
    if requested_id and db is not None:
        audit = db.get_audit(requested_id) or _find_by_id_prefix(db, requested_id)
        if audit:
            from rag.ingestion.audit_loader import audit_record_to_document
            text, metadata = audit_record_to_document(audit)
            chunk = RetrievedChunk(text=text, score=1.0, metadata=metadata)
            return AskControlPlaneAnswer(
                answer=text, citations=[chunk], status="SUCCESS", confidence=1.0,
            )
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

    answer_text, generation_mode, citation_check = synthesize_answer(question, chunks)
    confidence = round(sum(c.score for c in chunks[:3]) / min(3, len(chunks)), 3)

    if citation_check and not citation_check["ok"]:
        logger.warning(
            "Ask ControlPlane answer has invalid citations %s -- answer returned but review recommended.",
            citation_check["invalid_citations"],
        )

    return AskControlPlaneAnswer(
        answer=answer_text,
        citations=chunks,
        status="SUCCESS",
        confidence=confidence,
        generation_mode=generation_mode,
    )


def _find_by_id_prefix(db, prefix: str):
    """Scan recent audits for a prefix match on the full UUID."""
    for audit in db.recent_audits(limit=500):
        if audit["request_id"].startswith(prefix):
            return audit
    return None
