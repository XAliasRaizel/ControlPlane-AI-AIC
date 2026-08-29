"""
tests/test_agent_governance.py

Pure-stdlib unit tests for the agentic tool-call governance layer.
Written with unittest so they run with either `python -m unittest`
or your existing pytest suite (pytest auto-discovers
unittest.TestCase subclasses) without adding a new dependency.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

try:
    from backend.agents.governance import ToolGovernor  # noqa: E402
    from backend.agents.tools import _reset_world_state_for_tests, get_world_state_snapshot  # noqa: E402
except ModuleNotFoundError:
    from agents.governance import ToolGovernor  # noqa: E402
    from agents.tools import _reset_world_state_for_tests, get_world_state_snapshot  # noqa: E402

POLICY_PATH = Path(__file__).resolve().parents[1] / "policies" / "agent_tools.yaml"


def make_governor(session_risk: float = 0.0) -> ToolGovernor:
    return ToolGovernor(
        policy_path=str(POLICY_PATH),
        session_risk_lookup=lambda session_id: session_risk,
    )


class GovernanceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        _reset_world_state_for_tests()


class TestRefundGovernance(GovernanceTestCase):
    def test_small_refund_allows(self):
        gov = make_governor()
        result = gov.invoke(tool="issue_refund", role="support_agent", application="support-agent",
                             session_id="s1", args={"order_id": "o1", "amount": 100})
        self.assertEqual(result["decision"], "ALLOW")
        self.assertTrue(result["executed"])

    def test_mid_refund_requires_review_then_executes_on_approval(self):
        gov = make_governor()
        result = gov.invoke(tool="issue_refund", role="support_agent", application="support-agent",
                             session_id="s1", args={"order_id": "o2", "amount": 1200})
        self.assertEqual(result["decision"], "HUMAN_REVIEW")
        self.assertFalse(result["executed"])
        self.assertIsNotNone(result["pending_id"])

        approval = gov.resolve_pending(result["pending_id"], approve=True, resolved_by="mgr")
        self.assertTrue(approval["ok"])
        self.assertEqual(approval["status"], "APPROVED")
        refunds = get_world_state_snapshot()["issued_refunds"]
        self.assertTrue(any(r["order_id"] == "o2" for r in refunds))

    def test_large_refund_blocks_outright_even_for_a_manager(self):
        gov = make_governor()
        result = gov.invoke(tool="issue_refund", role="manager", application="support-agent",
                             session_id="s1", args={"order_id": "o3", "amount": 9000})
        self.assertEqual(result["decision"], "BLOCK")
        self.assertFalse(result["executed"])

    def test_rejected_pending_call_never_executes(self):
        gov = make_governor()
        result = gov.invoke(tool="issue_refund", role="support_agent", application="support-agent",
                             session_id="s1", args={"order_id": "o4", "amount": 1200})
        gov.resolve_pending(result["pending_id"], approve=False, resolved_by="mgr")
        refunds = get_world_state_snapshot()["issued_refunds"]
        self.assertFalse(any(r["order_id"] == "o4" for r in refunds))

    def test_unauthorized_role_still_needs_review_below_hard_cap(self):
        # An intern requesting a $150 refund: amount alone would ALLOW,
        # but issue_refund requires at least support_agent rank, so
        # authorization risk (severity floor) should push this to
        # HUMAN_REVIEW instead of a silent ALLOW.
        gov = make_governor()
        result = gov.invoke(tool="issue_refund", role="intern", application="support-agent",
                             session_id="s1", args={"order_id": "o5", "amount": 150})
        self.assertIn(result["decision"], ("HUMAN_REVIEW", "BLOCK"))
        self.assertFalse(result["executed"])


class TestDeleteRecordGovernance(GovernanceTestCase):
    def test_intern_cannot_delete_pii_record_even_with_review(self):
        gov = make_governor()
        result = gov.invoke(tool="delete_record", role="intern", application="hr-agent",
                             session_id="s2", args={"record_id": "cust_1001"})
        self.assertEqual(result["decision"], "BLOCK")
        self.assertFalse(result["executed"])

    def test_admin_can_delete_pii_record(self):
        gov = make_governor()
        result = gov.invoke(tool="delete_record", role="admin", application="hr-agent",
                             session_id="s2", args={"record_id": "cust_1002"})
        self.assertEqual(result["decision"], "ALLOW")
        self.assertTrue(result["executed"])

    def test_non_admin_manager_needs_review_not_auto_allow(self):
        gov = make_governor()
        result = gov.invoke(tool="delete_record", role="manager", application="hr-agent",
                             session_id="s2", args={"record_id": "cust_1001"})
        self.assertEqual(result["decision"], "HUMAN_REVIEW")


class TestEmailGovernance(GovernanceTestCase):
    def test_internal_email_without_pii_allows(self):
        gov = make_governor()
        result = gov.invoke(tool="send_email", role="employee", application="internal-copilot",
                             session_id="s3", args={"to": "teammate@corp.internal", "subject": "Hi",
                                                     "body": "See you at 3pm."})
        self.assertEqual(result["decision"], "ALLOW")

    def test_external_email_with_pii_blocks(self):
        gov = make_governor()
        result = gov.invoke(tool="send_email", role="employee", application="internal-copilot",
                             session_id="s3", args={"to": "someone@external.com", "subject": "Data",
                                                     "body": "Contact them at rahul@example.com"})
        self.assertEqual(result["decision"], "BLOCK")


class TestSessionCarryover(GovernanceTestCase):
    def test_elevated_session_risk_escalates_an_otherwise_clean_call(self):
        gov = make_governor(session_risk=0.75)
        result = gov.invoke(tool="send_email", role="employee", application="internal-copilot",
                             session_id="s4", args={"to": "teammate@corp.internal", "subject": "Notes",
                                                     "body": "Summary attached."})
        self.assertEqual(result["decision"], "HUMAN_REVIEW")

    def test_low_session_risk_does_not_escalate(self):
        gov = make_governor(session_risk=0.1)
        result = gov.invoke(tool="send_email", role="employee", application="internal-copilot",
                             session_id="s5", args={"to": "teammate@corp.internal", "subject": "Notes",
                                                     "body": "Summary attached."})
        self.assertEqual(result["decision"], "ALLOW")


class TestUnknownTool(GovernanceTestCase):
    def test_unknown_tool_is_blocked_not_crashed(self):
        gov = make_governor()
        result = gov.invoke(tool="wire_transfer_to_offshore_account", role="admin",
                             application="finance-agent", session_id="s6", args={})
        self.assertEqual(result["decision"], "BLOCK")
        self.assertFalse(result["executed"])


if __name__ == "__main__":
    unittest.main()
