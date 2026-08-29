"""
backend/app/agents/risk.py

Tool-call risk scoring, deliberately built in the same shape as the
request-level Risk Engine described in your architecture docs:

  1. Independent risk factors are computed per call (authorization,
     magnitude, sensitivity, session carryover).
  2. A *severity floor* -- the worst single factor -- puts a hard
     lower bound on the score, so one serious issue (e.g. an
     unauthorized caller) can't be diluted by several harmless ones.
  3. A weighted blend gives a smoother, more graded signal for calls
     that aren't clearly one thing or another.
  4. A reversibility multiplier scales the result up for actions that
     cannot be undone -- the tool-call analogue of the "critical
     application" context multiplier in the main Risk Engine.

Important: this does NOT trust the tool's own self-reported
risk_context for the fields that matter most. Authorization is
re-derived here from the caller's role, independent of anything the
tool says about itself, so a tool implementation that "forgets" to
flag its own risk can't quietly let something through.
"""
from __future__ import annotations

from typing import Any, Dict

from .models import ToolCallContext, ToolRiskSignal
from .tools import TOOL_REGISTRY

# Role hierarchy for this demo -- point this at your real RBAC/identity
# source before this goes anywhere near production.
_ROLE_RANK = {"intern": 0, "employee": 1, "support_agent": 1, "manager": 2, "admin": 3}


def _role_rank(role: str) -> int:
    return _ROLE_RANK.get(role, 0)  # unknown role -> least privileged, not most


def _authorization_risk(ctx: ToolCallContext) -> float:
    """0.0 = clearly authorized, 1.0 = clearly not."""
    if ctx.tool == "delete_record":
        return 0.0 if _role_rank(ctx.role) >= _ROLE_RANK["admin"] else 1.0
    if ctx.tool == "issue_refund":
        return 0.0 if _role_rank(ctx.role) >= _ROLE_RANK["support_agent"] else 1.0
    if ctx.tool == "send_email":
        return 0.0  # everyone can send email; magnitude/sensitivity carry the risk instead
    return 0.5  # unknown tool -- treat cautiously rather than assume safe


def _magnitude_risk(ctx: ToolCallContext) -> float:
    if ctx.tool == "issue_refund":
        amount = float(ctx.args.get("amount", 0))
        # smooth ramp: $0 -> 0.0, $500 -> 0.25, $2000+ -> 1.0 (capped)
        return max(0.0, min(1.0, amount / 2000.0))
    return 0.0


def _sensitivity_risk(ctx: ToolCallContext, risk_context: Dict[str, Any]) -> float:
    if ctx.tool == "delete_record":
        return 0.9 if risk_context.get("record_contains_pii") else 0.2
    if ctx.tool == "send_email":
        external = risk_context.get("is_external_recipient", False)
        pii = risk_context.get("contains_pii", False)
        if external and pii:
            return 0.95
        if pii:
            return 0.5
        if external:
            return 0.2
        return 0.0
    return 0.0


def _reversibility_multiplier(ctx: ToolCallContext) -> float:
    spec = TOOL_REGISTRY.get(ctx.tool)
    return 1.0 if (spec is None or spec.reversible) else 1.15


def score_tool_call(ctx: ToolCallContext) -> ToolRiskSignal:
    spec = TOOL_REGISTRY.get(ctx.tool)
    risk_context = spec.describe_risk(ctx.args) if spec else {}

    auth = _authorization_risk(ctx)
    magnitude = _magnitude_risk(ctx)
    sensitivity = _sensitivity_risk(ctx, risk_context)
    session = min(1.0, max(0.0, ctx.session_risk))

    severity_floor = max(auth, magnitude, sensitivity, session)
    blended = 0.35 * auth + 0.30 * magnitude + 0.20 * sensitivity + 0.15 * session
    base = max(severity_floor, blended)
    overall = min(1.0, base * _reversibility_multiplier(ctx))

    evidence = []
    if auth >= 1.0:
        evidence.append(f"role '{ctx.role}' is not authorized to call {ctx.tool}")
    if magnitude >= 0.5:
        evidence.append(f"amount {ctx.args.get('amount')} exceeds the normal single-call threshold")
    if sensitivity >= 0.5:
        evidence.append("target is flagged as sensitive / contains PII")
    if session > 0.3:
        evidence.append(f"session already carries elevated risk ({session:.2f})")

    return ToolRiskSignal(
        tool=ctx.tool,
        score=round(overall, 4),
        confidence=0.95 if spec else 0.5,
        factors={
            "authorization": auth, "magnitude": magnitude,
            "sensitivity": sensitivity, "session_carryover": session,
            "reversibility_multiplier": _reversibility_multiplier(ctx),
        },
        raw_context=risk_context,
        evidence=evidence,
    )
