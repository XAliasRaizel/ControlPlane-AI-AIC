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

UPGRADE: Live Operational Data Context
  The chatbot now answers operational questions by querying the live
  governance audit database directly:
  - "Why was request X blocked?"     → fetches audit record from DB
  - "What policy blocked most today?" → queries blocked_by_policy metrics
  - "What is the current risk level?" → queries risk_trend from DB
  - "Show me recent decisions"        → last 5 governance decisions

  This data is injected as high-priority RetrievedChunk items that rank
  above static policy documents, ensuring answers about live system state
  are grounded in actual governance logs, not hallucinated from training.

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

# Intent patterns for live operational queries
_METRICS_INTENTS = re.compile(
    r"\b(metrics|stats|statistics|dashboard|overview|summary|how many|total|count)\b", re.I
)
_BLOCK_POLICY_INTENTS = re.compile(
    r"\b(blocked|block|blocked most|top policy|which policy|most blocked|top rule)\b", re.I
)
_RISK_INTENTS = re.compile(
    r"\b(risk|risk level|risk trend|high risk|recent risk|current risk)\b", re.I
)
_RECENT_DECISIONS_INTENTS = re.compile(
    r"\b(recent|latest|last|decisions|requests|audit|history)\b", re.I
)


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
        provider=rag_settings.llm_provider,
        ollama_model=rag_settings.ollama_model,
        ollama_host=rag_settings.ollama_host,
    )



# ---------------------------------------------------------------------------
# Live Operational Data Context (Change 4)
# ---------------------------------------------------------------------------

def _get_live_operational_context(question: str, db) -> list[RetrievedChunk]:
    """Query the live governance audit database and return RetrievedChunk items
    for questions about live system state. These rank above static policy docs.
    Returns empty list if db is None or query doesn't match an operational intent.
    """
    if db is None:
        return []

    live_chunks: list[RetrievedChunk] = []

    # --- Platform-wide metrics ---
    if _METRICS_INTENTS.search(question):
        try:
            m = db.richer_metrics()
            text = (
                f"[LIVE PLATFORM METRICS]\n"
                f"Total Requests: {m.get('total_requests', 0)}\n"
                f"Blocked: {m.get('blocked', 0)}  |  Allowed: {m.get('allowed', 0)}  "
                f"|  Human Review: {m.get('human_review', 0)}\n"
                f"Avg Latency: {m.get('avg_latency_ms', 0):.1f}ms\n"
                f"Risk Distribution — Low: {m.get('risk_distribution', {}).get('low', 0)}  "
                f"Medium: {m.get('risk_distribution', {}).get('medium', 0)}  "
                f"High: {m.get('risk_distribution', {}).get('high', 0)}\n"
                f"Total Feedback: {m.get('feedback_count', 0)}"
            )
            live_chunks.append(RetrievedChunk(
                text=text, score=1.0,
                metadata={"document_type": "live_metrics", "source": "governance_db"},
            ))
        except Exception as exc:
            logger.warning("Failed to fetch live metrics for chatbot: %s", exc)

    # --- Most blocked policy rules ---
    if _BLOCK_POLICY_INTENTS.search(question):
        try:
            m = db.richer_metrics()
            blocked_by_policy = m.get("blocked_by_policy", {})
            if blocked_by_policy:
                sorted_rules = sorted(blocked_by_policy.items(), key=lambda x: x[1], reverse=True)
                lines = [f"  {rule}: {count} blocks" for rule, count in sorted_rules[:5]]
                text = "[LIVE BLOCK ANALYSIS]\nTop policies triggering BLOCK decisions:\n" + "\n".join(lines)
                live_chunks.append(RetrievedChunk(
                    text=text, score=0.99,
                    metadata={"document_type": "live_block_analysis", "source": "governance_db"},
                ))
        except Exception as exc:
            logger.warning("Failed to fetch block analysis for chatbot: %s", exc)

    # --- Recent risk trend ---
    if _RISK_INTENTS.search(question):
        try:
            m = db.richer_metrics()
            risk_trend = m.get("risk_trend", [])
            if risk_trend:
                recent = risk_trend[-5:]
                lines = [f"  {r['ts'][:19]}: risk={r['risk']}" for r in recent]
                text = "[LIVE RISK TREND — Last 5 Requests]\n" + "\n".join(lines)
                live_chunks.append(RetrievedChunk(
                    text=text, score=0.98,
                    metadata={"document_type": "live_risk_trend", "source": "governance_db"},
                ))
        except Exception as exc:
            logger.warning("Failed to fetch risk trend for chatbot: %s", exc)

    # --- Recent decisions ---
    if _RECENT_DECISIONS_INTENTS.search(question) and not _looks_like_request_id_question(question):
        try:
            recent = db.recent_audits(limit=5)
            if recent:
                lines = []
                for a in recent:
                    dd = a.get("decision_details", {})
                    action = dd.get("action", "?") if isinstance(dd, dict) else "?"
                    risk = a.get("risk", {})
                    risk_val = risk.get("overall_risk", "?") if isinstance(risk, dict) else "?"
                    lines.append(
                        f"  [{a['created_at'][:19]}] {action} | risk={risk_val} | "
                        f"id={a['request_id'][:12]}..."
                    )
                text = "[LIVE RECENT GOVERNANCE DECISIONS]\n" + "\n".join(lines)
                live_chunks.append(RetrievedChunk(
                    text=text, score=0.97,
                    metadata={"document_type": "live_recent_decisions", "source": "governance_db"},
                ))
        except Exception as exc:
            logger.warning("Failed to fetch recent decisions for chatbot: %s", exc)

    return live_chunks


# ---------------------------------------------------------------------------
# Core synthesis + ask entry point (unchanged API)
# ---------------------------------------------------------------------------

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

    UPGRADE: Now queries live operational audit data before falling back
    to static policy document retrieval, so questions like "what policy
    blocks the most?" and "what is the current risk level?" are answered
    from real governance data, not hallucinated from training.
    """
    if not question or not question.strip():
        return AskControlPlaneAnswer(
            answer="Please ask a specific question about a policy, decision, or audit record.",
            citations=[], status="INVALID_REQUEST", confidence=0.0,
        )

    # Fast-path: explicit request_id lookup from live DB
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

    # Inject live operational context FIRST (highest priority chunks)
    live_chunks = _get_live_operational_context(question, db)

    # Then fetch policy/regulatory context via hybrid retrieval
    try:
        policy_chunks = hybrid_retrieve(question, top_k=top_k)
    except Exception:
        policy_chunks = []

    # Merge: live data chunks rank at top (score=0.97–1.0), policy chunks below
    all_chunks = live_chunks + policy_chunks
    all_chunks = all_chunks[:top_k + len(live_chunks)]  # keep live chunks + top_k policy

    if not all_chunks:
        return AskControlPlaneAnswer(
            answer=_INSUFFICIENT_EVIDENCE_MESSAGE, citations=[],
            status="INSUFFICIENT_EVIDENCE", confidence=0.0,
        )

    answer_text, generation_mode, citation_check = synthesize_answer(question, all_chunks)
    confidence = round(sum(c.score for c in all_chunks[:3]) / min(3, len(all_chunks)), 3)

    if citation_check and not citation_check["ok"]:
        logger.warning(
            "Ask ControlPlane answer has invalid citations %s -- answer returned but review recommended.",
            citation_check["invalid_citations"],
        )

    return AskControlPlaneAnswer(
        answer=answer_text,
        citations=all_chunks,
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
