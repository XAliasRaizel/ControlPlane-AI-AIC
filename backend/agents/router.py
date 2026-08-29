"""
backend/app/agents/router.py

FastAPI + Pydantic wrapper around the (fully unit-tested) core in
governance.py / risk.py / policy.py / tools.py. Deliberately thin: all
the actual decision logic already lives -- and is already tested --
in the framework-agnostic core, so this file just translates HTTP
requests into `ToolGovernor.invoke(...)` calls and back.

NOTE ON TESTING: fastapi/pydantic are not installed in the sandbox
this feature was built and unit-tested in, so this specific file
could not be executed there. It has been written conservatively
against stable, well-documented FastAPI/Pydantic v2 APIs and kept as
a pure pass-through with no new business logic, to minimise the
surface area that couldn't be exercised directly. Please run your own
`pytest`/`uvicorn` smoke test on this one file after dropping it in --
everything it calls into (ToolGovernor and friends) is already green.

Mount it from your main FastAPI app, e.g. in backend/app/main.py:

    from app.agents.router import router as agent_governance_router
    app.include_router(agent_governance_router)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .governance import ToolGovernor

router = APIRouter(prefix="/agent", tags=["agent-tool-governance"])

# A single process-wide governor instance for the demo. In the real
# gateway, construct this once at app startup (see main.py) with real
# audit_sink / session_risk_lookup adapters wired to your existing
# audit.py and session/risk tracker, and inject it via FastAPI's
# dependency system instead of a module-level singleton.
_governor = ToolGovernor()


class ToolCallRequest(BaseModel):
    tool: str = Field(..., description="Registered tool name, e.g. 'issue_refund'")
    args: Dict[str, Any] = Field(default_factory=dict)
    role: str = Field(..., description="Caller's role, e.g. 'support_agent', 'admin'")
    application: str = Field(..., description="Calling application/agent id")
    session_id: str
    request_id: Optional[str] = None


class ToolCallResponse(BaseModel):
    decision: str
    reason: str
    matched_rule: Optional[str] = None
    risk: float
    evidence: list[str]
    executed: bool
    result: Optional[Any] = None
    pending_id: Optional[str] = None
    request_id: Optional[str] = None


class ResolvePendingRequest(BaseModel):
    approve: bool
    resolved_by: str


@router.post("/act", response_model=ToolCallResponse)
def act(payload: ToolCallRequest) -> ToolCallResponse:
    result = _governor.invoke(
        tool=payload.tool, args=payload.args, role=payload.role,
        application=payload.application, session_id=payload.session_id,
        request_id=payload.request_id,
    )
    return ToolCallResponse(**result)


@router.get("/pending")
def list_pending() -> list[dict]:
    return [
        {
            "pending_id": p.pending_id,
            "tool": p.context.tool,
            "role": p.context.role,
            "args": p.context.args,
            "risk": p.risk_signal.score,
            "reason": p.decision.reason,
            "created_at": p.created_at,
        }
        for p in _governor.queue.list_pending()
    ]


@router.post("/pending/{pending_id}/resolve")
def resolve_pending(pending_id: str, payload: ResolvePendingRequest) -> dict:
    outcome = _governor.resolve_pending(pending_id, approve=payload.approve, resolved_by=payload.resolved_by)
    if not outcome.get("ok"):
        raise HTTPException(status_code=404, detail=outcome.get("error", "not found"))
    return outcome
