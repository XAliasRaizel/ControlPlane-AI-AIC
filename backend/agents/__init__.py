"""
Agentic tool-call governance for ControlPlane.ai.

This subpackage adds a governance layer for AI *agents* that take
actions (send email, issue refunds, delete records, ...), not just
generate text. It reuses the same mental model as the rest of
ControlPlane -- score risk, apply policy, decide, audit -- so it reads
like a natural extension of the existing hot-path/risk/policy/decision
engines rather than a bolted-on side project.

Public entry point: ToolGovernor (see governance.py).
"""
from .governance import ToolGovernor
from .models import GovernanceDecision, PendingToolCall, ToolCallContext, ToolRiskSignal

__all__ = [
    "ToolGovernor",
    "GovernanceDecision",
    "PendingToolCall",
    "ToolCallContext",
    "ToolRiskSignal",
]
