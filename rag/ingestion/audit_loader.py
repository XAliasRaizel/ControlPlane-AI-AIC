"""Loads governance audit records into the Ask ControlPlane retrieval
corpus -- STRICT ALLOW-LIST ONLY (spec Section 4 security requirement).

Never pulls raw prompts or response text. `build_audit_context()`
(backend/audit/store.py) already produces an allow-listed context with no
raw prompt and only an HMAC fingerprint instead of the raw user id -- this
loader is a second, independent allow-list on top of that, deliberately
redundant with it, so a future change to build_audit_context() can't
silently widen what ends up in this corpus without this file also being
touched.
"""

from __future__ import annotations

from rag.schemas import Chunk

# Every field this loader is willing to read out of an audit record. Add
# to this list deliberately, not by widening a wildcard.
_ALLOWED_CONTEXT_FIELDS = {
    "application_id", "department", "user_role", "data_classification",
    "model", "provider", "tool_count",
}


def _risk_level(score: float) -> str:
    if score >= 0.8:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"


def audit_record_to_document(audit: dict) -> tuple[str, dict]:
    """One governance_audits row -> (prose_text, metadata). Assumes `audit`
    is already the parsed dict shape returned by Database.get_audit() /
    recent_audits() (JSON fields already loaded).
    """
    request_id = audit["request_id"]
    ctx = {k: v for k, v in (audit.get("audit_context") or {}).items() if k in _ALLOWED_CONTEXT_FIELDS}
    risk = audit.get("risk") or {}
    policy = audit.get("policy") or {}
    decision = audit.get("decision_details") or {}
    detectors = audit.get("detector_results") or []

    risk_score = float(risk.get("overall_risk", 0.0))
    action = decision.get("action", "UNKNOWN")
    reason = decision.get("reason", "")
    policy_id = decision.get("policy_id") or policy.get("policy_id", "")

    # Pattern-level labels only (e.g. "pii: PII_DETECTED") -- detector
    # evidence lists contain matched *pattern names*, never raw matched
    # text (see backend/detectors/pii.py) -- safe to surface.
    detector_summary = ", ".join(
        f"{d.get('detector_name')}: {d.get('label')} (score {d.get('score')})"
        for d in detectors
    )

    lines = [
        f"Governance audit record {request_id}, logged {audit.get('created_at', 'unknown time')}.",
        f"Application: {ctx.get('application_id', 'unknown')}, department: {ctx.get('department', 'unknown')}, "
        f"caller role: {ctx.get('user_role', 'unknown')}, data classification: {ctx.get('data_classification', 'unspecified')}.",
        f"Decision: {action}. Risk score: {round(risk_score, 3)} ({_risk_level(risk_score)}). "
        f"Matched policy: {policy_id or 'none'}.",
        f"Decision reason: {reason}" if reason else "",
        f"Detector results: {detector_summary}" if detector_summary else "",
    ]
    text = "\n".join(line for line in lines if line)

    metadata = {
        "source": "audit_log",
        "document_type": "audit_record",
        "request_id": request_id,
        "decision": action,
        "risk_level": _risk_level(risk_score),
        "policy_id": policy_id or "",
        "application_id": ctx.get("application_id", ""),
    }
    return text, metadata


def load_audit_corpus(audits: list[dict]) -> list[Chunk]:
    """`audits` is the output of Database.recent_audits()."""
    chunks: list[Chunk] = []
    for audit in audits:
        try:
            text, metadata = audit_record_to_document(audit)
        except Exception as exc:
            import logging
            logging.getLogger("controlplane.rag").warning(
                "Skipping malformed audit record %s: %s", audit.get("request_id", "?"), exc
            )
            continue
        chunks.append(Chunk(chunk_id=f"audit::{audit['request_id']}", text=text, metadata=metadata))
    return chunks
