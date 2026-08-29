"""
backend/agents/adapters.py

Wires ToolGovernor's two injection points (audit_sink, session_risk_lookup)
into ControlPlane's real infrastructure. governance.py is intentionally
untouched by this -- it only ever depends on plain callables passed into
its constructor, so this adapter file is the entire integration surface.

Session risk: no multi-turn/session risk tracker exists yet in this repo
(that's the Round 2 board's CUSUM item, separately scoped). The lookup
below is an honest stand-in that always returns 0.0, exactly like the
placeholder it replaces. Swap only the body of _lookup_session_risk once
a real tracker exists; ToolGovernor calls it as session_risk_lookup(session_id)
and doesn't need to change.
"""
from __future__ import annotations

from backend.audit.store import Database

from .governance import ToolGovernor


def build_tool_governor(db: Database, policy_path: str | None = None) -> ToolGovernor:
    def _audit_sink(record: dict) -> None:
        db.save_tool_call_event(record)

    def _lookup_session_risk(session_id: str) -> float:
        # TODO(session-risk): replace once a session/CUSUM tracker lands.
        return 0.0

    return ToolGovernor(
        policy_path=policy_path,
        audit_sink=_audit_sink,
        session_risk_lookup=_lookup_session_risk,
    )
