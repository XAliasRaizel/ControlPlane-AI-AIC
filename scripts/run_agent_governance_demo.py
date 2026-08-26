#!/usr/bin/env python3
"""
scripts/run_agent_governance_demo.py

Standalone, dependency-free (stdlib + PyYAML only) walkthrough of the
agentic tool-call governance layer. Mirrors the existing golden-path
demo pattern -- run it, read the trace, use it as a live-demo script.

    python3 scripts/run_agent_governance_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

try:
    from backend.agents.governance import ToolGovernor  # noqa: E402
    from backend.agents.tools import get_world_state_snapshot  # noqa: E402
except ModuleNotFoundError:
    from agents.governance import ToolGovernor  # noqa: E402
    from agents.tools import get_world_state_snapshot  # noqa: E402

POLICY_PATH = Path(__file__).resolve().parents[1] / "policies" / "agent_tools.yaml"

# Simulated session-risk store. In the real system this is a lookup into
# whatever tracks cumulative per-session risk (e.g. a CUSUM tracker fed by
# the hot-path detectors) -- here it's just a dict we set by hand so the
# "compounding risk" scenario is reproducible.
SESSION_RISK: dict[str, float] = {}


def _print_trace(label: str, result: dict) -> None:
    print(f"\n--- {label} ---")
    print(f"  decision      : {result['decision']}")
    print(f"  matched_rule  : {result['matched_rule']}")
    print(f"  risk          : {result['risk']}")
    print(f"  reason        : {result['reason']}")
    if result["evidence"]:
        print(f"  evidence      : {result['evidence']}")
    print(f"  executed      : {result['executed']}")
    if result["pending_id"]:
        print(f"  pending_id    : {result['pending_id']}")
    if result["result"]:
        print(f"  tool result   : {result['result']}")


def main() -> None:
    audit_log: list[dict] = []
    governor = ToolGovernor(
        policy_path=str(POLICY_PATH),
        audit_sink=lambda record: audit_log.append(record),
        session_risk_lookup=lambda session_id: SESSION_RISK.get(session_id, 0.0),
    )

    print("=" * 72)
    print("ControlPlane.ai -- Agentic Tool-Call Governance: golden path demo")
    print("=" * 72)

    r1 = governor.invoke(
        tool="send_email", role="support_agent", application="support-agent",
        session_id="sess-1", args={"to": "teammate@corp.internal", "subject": "Order update",
                                    "body": "Your order has shipped, thanks for your patience!"},
    )
    _print_trace("1. Internal email, no PII -> should ALLOW", r1)

    r2 = governor.invoke(
        tool="issue_refund", role="support_agent", application="support-agent",
        session_id="sess-1", args={"order_id": "ord_501", "amount": 120},
    )
    _print_trace("2. Small refund ($120) -> should ALLOW", r2)

    r3 = governor.invoke(
        tool="issue_refund", role="support_agent", application="support-agent",
        session_id="sess-1", args={"order_id": "ord_502", "amount": 1500},
    )
    _print_trace("3. Mid-size refund ($1500) -> should HUMAN_REVIEW", r3)

    if r3["pending_id"]:
        approval = governor.resolve_pending(r3["pending_id"], approve=True, resolved_by="manager_priya")
        print(f"  -> manager approves: {approval}")

    r4 = governor.invoke(
        tool="issue_refund", role="support_agent", application="support-agent",
        session_id="sess-1", args={"order_id": "ord_503", "amount": 5000},
    )
    _print_trace("4. Large refund ($5000) -> should BLOCK outright", r4)

    r5 = governor.invoke(
        tool="delete_record", role="intern", application="hr-agent",
        session_id="sess-2", args={"record_id": "cust_1001"},
    )
    _print_trace("5. Intern deleting a PII customer record -> should BLOCK", r5)

    r6 = governor.invoke(
        tool="delete_record", role="admin", application="hr-agent",
        session_id="sess-2", args={"record_id": "cust_1002"},
    )
    _print_trace("6. Admin deleting the same kind of record -> should ALLOW", r6)

    # Connective scenario: a session that already picked up elevated risk
    # earlier in the conversation (e.g. a multi-turn probe your session-risk
    # tracker flagged) makes a *normally fine* tool call escalate, even
    # though nothing about this specific call looks unusual on its own.
    SESSION_RISK["sess-3"] = 0.72
    r7 = governor.invoke(
        tool="send_email", role="employee", application="internal-copilot",
        session_id="sess-3", args={"to": "teammate@corp.internal", "subject": "Notes",
                                    "body": "Here's a summary of today's meeting."},
    )
    _print_trace("7. Ordinary internal email, but session risk is elevated -> should HUMAN_REVIEW", r7)

    r8 = governor.invoke(
        tool="wire_the_whole_treasury", role="admin", application="finance-agent",
        session_id="sess-4", args={},
    )
    _print_trace("8. A tool that isn't even registered -> should BLOCK, not crash", r8)

    print("\n" + "=" * 72)
    print("World state after the run:")
    print(json.dumps(get_world_state_snapshot(), indent=2, default=str))
    print(f"\nAudit trail entries captured: {len(audit_log)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
