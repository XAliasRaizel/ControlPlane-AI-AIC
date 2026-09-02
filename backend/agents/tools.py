"""
backend/app/agents/tools.py

Toy tool implementations for the agentic tool-call governance demo.
These simulate side effects -- they never call a real mail server,
payment gateway, or database -- so the whole thing is safe to run
anywhere, including your CI, with zero external services.

Design note: each tool exposes a `describe_risk(args)` function that
runs *before* execution and reports the raw facts the governor needs
(is the recipient external, does the record hold PII, ...). risk.py
re-derives the factors that matter most (authorization, amount)
independently of this self-report, so a tool that understates its own
risk can't quietly slip a call through.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple

# ---------------------------------------------------------------------------
# In-memory "world state" so the demo has something concrete to show
# changing. Swap for real integrations (SES, a payment gateway, your CRM)
# when this graduates past the prototype stage.
# ---------------------------------------------------------------------------

_SENT_EMAILS: list[dict] = []
_ISSUED_REFUNDS: list[dict] = []
_DELETED_RECORDS: Dict[str, dict] = {}

_DEFAULT_RECORD_STORE = {
    "cust_1001": {"type": "customer", "pii": True, "note": "Rahul Sharma, HR record"},
    "cust_1002": {"type": "customer", "pii": True, "note": "Aryan Verma, HR record"},
    "log_9001": {"type": "log", "pii": False, "note": "system log line"},
}
_FAKE_RECORD_STORE: Dict[str, dict] = dict(_DEFAULT_RECORD_STORE)

_PII_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\b\d{10}\b|\b\d{3}-\d{2}-\d{4}\b")

_INTERNAL_DOMAINS = ("controlplane.ai", "corp.internal")


def _is_external_domain(email: str) -> bool:
    domain = email.split("@")[-1].lower() if "@" in email else email.lower()
    return domain not in _INTERNAL_DOMAINS


def _reset_world_state_for_tests() -> None:
    """Test-only helper. Clears in-memory state between test cases so
    tests don't leak side effects into each other."""
    _SENT_EMAILS.clear()
    _ISSUED_REFUNDS.clear()
    _DELETED_RECORDS.clear()
    _FAKE_RECORD_STORE.clear()
    _FAKE_RECORD_STORE.update(_DEFAULT_RECORD_STORE)


@dataclass
class ToolCallOutcome:
    tool: str
    executed: bool
    result: Any
    risk_context: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------

def send_email_describe_risk(args: dict) -> dict:
    to = args.get("to", "")
    body = args.get("body", "")
    return {
        "is_external_recipient": _is_external_domain(to),
        "contains_pii": bool(_PII_PATTERN.search(body)),
        "recipient": to,
    }


def send_email_execute(args: dict) -> ToolCallOutcome:
    to, subject, body = args["to"], args.get("subject", ""), args.get("body", "")
    risk_context = send_email_describe_risk(args)
    message_id = f"msg_{uuid.uuid4().hex[:8]}"
    _SENT_EMAILS.append({"id": message_id, "to": to, "subject": subject, "body": body, "ts": time.time()})
    return ToolCallOutcome(
        tool="send_email", executed=True,
        result={"message_id": message_id, "to": to, "subject": subject},
        risk_context=risk_context,
    )


# ---------------------------------------------------------------------------
# issue_refund
# ---------------------------------------------------------------------------

def issue_refund_describe_risk(args: dict) -> dict:
    return {"amount": float(args.get("amount", 0))}


def issue_refund_execute(args: dict) -> ToolCallOutcome:
    order_id, amount = args["order_id"], float(args["amount"])
    risk_context = issue_refund_describe_risk(args)
    refund_id = f"rfnd_{uuid.uuid4().hex[:8]}"
    _ISSUED_REFUNDS.append({"id": refund_id, "order_id": order_id, "amount": amount, "ts": time.time()})
    return ToolCallOutcome(
        tool="issue_refund", executed=True,
        result={"refund_id": refund_id, "order_id": order_id, "amount": amount},
        risk_context=risk_context,
    )


# ---------------------------------------------------------------------------
# delete_record
# ---------------------------------------------------------------------------

def delete_record_describe_risk(args: dict) -> dict:
    record_id = args.get("record_id", "")
    # Unknown record id -> assume the worst (contains PII) rather than the best.
    record = _FAKE_RECORD_STORE.get(record_id, {"type": "unknown", "pii": True})
    return {
        "record_type": record["type"],
        "record_contains_pii": bool(record.get("pii", True)),
        "record_exists": record_id in _FAKE_RECORD_STORE,
    }


def delete_record_execute(args: dict) -> ToolCallOutcome:
    record_id = args["record_id"]
    # Snapshot risk context BEFORE mutating state -- otherwise a
    # post-delete describe_risk() call would (wrongly) report the
    # record as no longer existing.
    risk_context = delete_record_describe_risk(args)
    record = _FAKE_RECORD_STORE.pop(record_id, None)
    if record is not None:
        _DELETED_RECORDS[record_id] = record
    return ToolCallOutcome(
        tool="delete_record", executed=True,
        result={"record_id": record_id, "deleted": record is not None},
        risk_context=risk_context,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    describe_risk: Callable[[dict], dict]
    execute: Callable[[dict], ToolCallOutcome]
    required_args: Tuple[str, ...]
    reversible: bool  # feeds the reversibility multiplier in risk.py


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "send_email": ToolSpec(
        name="send_email", describe_risk=send_email_describe_risk, execute=send_email_execute,
        required_args=("to", "subject", "body"), reversible=False,  # you can't unsend an email
    ),
    "issue_refund": ToolSpec(
        name="issue_refund", describe_risk=issue_refund_describe_risk, execute=issue_refund_execute,
        required_args=("order_id", "amount"), reversible=True,  # can be clawed back administratively
    ),
    "delete_record": ToolSpec(
        name="delete_record", describe_risk=delete_record_describe_risk, execute=delete_record_execute,
        required_args=("record_id",), reversible=False,  # treated as irreversible for this demo
    ),
}


def get_world_state_snapshot() -> dict:
    """Demo/debug helper so the golden-path script can show what actually changed."""
    return {
        "sent_emails": list(_SENT_EMAILS),
        "issued_refunds": list(_ISSUED_REFUNDS),
        "deleted_records": dict(_DELETED_RECORDS),
        "remaining_records": dict(_FAKE_RECORD_STORE),
    }
