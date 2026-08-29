"""
backend/app/agents/governance.py

ToolGovernor is the single choke point every tool call must pass
through. This mirrors the "interceptor" pattern used by production
agent frameworks (LangChain's HITL middleware, Microsoft Agent
Framework's DelegatingChatClient, Amazon Bedrock Agents' user
confirmation step): the agent *proposes* a call, the governor
*decides*, and only the governor -- never the agent -- is allowed to
actually run the tool.

That separation is what makes the gate meaningful. An agent that has
been prompt-injected into "ignoring its instructions" still cannot
execute a blocked or pending call, because it never held the ability
to execute anything in the first place -- it can only ask.

Integration note: `audit_sink` and `session_risk_lookup` are injected
so this module has zero hard dependency on your existing audit.py /
risk_engine.py. Pass in adapters that call your real implementations;
the defaults below are safe no-ops / stand-ins so this file runs
completely standalone until you wire it in. See
docs/agent_tool_governance_spec.md for the wiring guide.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, Optional

from .models import GovernanceDecision, ToolCallContext, ToolRiskSignal
from .policy import PolicyRule, evaluate, load_rules
from .queue import ReviewQueue
from .risk import score_tool_call
from .tools import TOOL_REGISTRY

AuditSink = Callable[[Dict[str, Any]], None]
SessionRiskLookup = Callable[[str], float]

# Fallback thresholds used only when no explicit policy rule matches --
# policy always wins over the raw score when a rule does match (see
# _decide below), same as "RISK -> POLICY -> DECISION, not RISK -> DECISION"
# in the main architecture.
BLOCK_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.50


def _noop_audit_sink(record: Dict[str, Any]) -> None:
    # Replace with backend.app.audit.record_event(...) (or your
    # equivalent) once this is wired into the real audit store.
    pass


def _zero_session_risk(session_id: str) -> float:
    # Replace with a lookup into your session/CUSUM risk tracker.
    return 0.0


class ToolGovernor:
    def __init__(
        self,
        policy_path: Optional[str] = None,
        audit_sink: AuditSink = _noop_audit_sink,
        session_risk_lookup: SessionRiskLookup = _zero_session_risk,
    ) -> None:
        self._rules = load_rules(policy_path)
        self._queue = ReviewQueue()
        self._audit_sink = audit_sink
        self._session_risk_lookup = session_risk_lookup

    @property
    def queue(self) -> ReviewQueue:
        return self._queue

    def invoke(self, tool: str, args: Dict[str, Any], role: str, application: str,
               session_id: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        request_id = request_id or f"req_{uuid.uuid4().hex[:8]}"

        if tool not in TOOL_REGISTRY:
            decision = GovernanceDecision(action="BLOCK", reason=f"unknown tool '{tool}'", risk=1.0)
            return self._finalize(None, None, decision, executed=False, result=None, request_id=request_id, tool=tool)

        ctx = ToolCallContext(
            session_id=session_id, role=role, application=application, tool=tool,
            args=args, session_risk=self._session_risk_lookup(session_id), request_id=request_id,
        )
        risk = score_tool_call(ctx)
        rule = evaluate(ctx, risk, self._rules)
        decision = self._decide(risk, rule)

        if decision.action == "BLOCK":
            return self._finalize(ctx, risk, decision, executed=False, result=None)

        if decision.action == "HUMAN_REVIEW":
            pending = self._queue.enqueue(ctx, risk, decision)
            return self._finalize(ctx, risk, decision, executed=False, result=None, pending_id=pending.pending_id)

        # ALLOW
        outcome = TOOL_REGISTRY[tool].execute(args)
        return self._finalize(ctx, risk, decision, executed=True, result=outcome.result)

    def resolve_pending(self, pending_id: str, approve: bool, resolved_by: str) -> Dict[str, Any]:
        record = self._queue.resolve(pending_id, approve=approve, resolved_by=resolved_by)
        if record is None:
            return {"ok": False, "error": "no such pending call, or it was already resolved"}
        if not approve:
            self._audit_sink({"event": "tool_call_rejected", "pending_id": pending_id, "resolved_by": resolved_by})
            return {"ok": True, "status": "REJECTED"}
        outcome = TOOL_REGISTRY[record.context.tool].execute(record.context.args)
        self._audit_sink({
            "event": "tool_call_approved_and_executed", "pending_id": pending_id,
            "resolved_by": resolved_by, "tool": record.context.tool, "result": outcome.result,
        })
        return {"ok": True, "status": "APPROVED", "result": outcome.result}

    def _decide(self, risk: ToolRiskSignal, rule: Optional[PolicyRule]) -> GovernanceDecision:
        if rule is not None:
            # Policy always wins over the raw score when a rule matches --
            # e.g. an admin deleting a PII-flagged record scores high on
            # sensitivity, but an explicit policy can still ALLOW it because
            # context (the role) makes it legitimate. Risk alone never
            # makes that call.
            return GovernanceDecision(action=rule.action, reason=rule.reason, risk=risk.score,
                                       matched_rule=rule.id, confidence=risk.confidence)
        if risk.score >= BLOCK_THRESHOLD:
            return GovernanceDecision(action="BLOCK", reason="risk score above the hard block threshold",
                                       risk=risk.score, confidence=risk.confidence)
        if risk.score >= REVIEW_THRESHOLD:
            return GovernanceDecision(action="HUMAN_REVIEW", reason="risk score requires sign-off",
                                       risk=risk.score, confidence=risk.confidence)
        return GovernanceDecision(action="ALLOW", reason="risk within normal operating range",
                                   risk=risk.score, confidence=risk.confidence)

    def _finalize(self, ctx: Optional[ToolCallContext], risk: Optional[ToolRiskSignal],
                  decision: GovernanceDecision, *, executed: bool, result: Any,
                  pending_id: Optional[str] = None, request_id: Optional[str] = None,
                  tool: Optional[str] = None) -> Dict[str, Any]:
        record = {
            "request_id": ctx.request_id if ctx else request_id,
            "tool": ctx.tool if ctx else tool,
            "role": ctx.role if ctx else None,
            "session_id": ctx.session_id if ctx else None,
            "decision": decision.action,
            "reason": decision.reason,
            "matched_rule": decision.matched_rule,
            "risk": decision.risk,
            "factors": risk.factors if risk else {},
            "evidence": risk.evidence if risk else [],
            "executed": executed,
            "pending_id": pending_id,
            "timestamp": time.time(),
        }
        self._audit_sink(record)
        return {
            "decision": decision.action,
            "reason": decision.reason,
            "matched_rule": decision.matched_rule,
            "risk": decision.risk,
            "evidence": risk.evidence if risk else [],
            "executed": executed,
            "result": result,
            "pending_id": pending_id,
            "request_id": ctx.request_id if ctx else request_id,
        }
