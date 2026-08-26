"""
backend/app/agents/policy.py

Loads policies/agent_tools.yaml and evaluates it against a tool call's
context + risk signal. Conditions are short boolean expressions
authored by your own team in a trusted YAML file -- NOT end-user
input -- evaluated through a restricted eval with an empty
__builtins__ table and a fixed set of allowed names. That keeps the
policy layer declarative without pulling in a full rules-engine
dependency you may not have installed yet.

Do not point `condition` strings at anything derived directly from
untrusted user text; they should only ever reference the fields
exposed in `_build_eval_context` below.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .models import ToolCallContext, ToolRiskSignal


@dataclass
class PolicyRule:
    id: str
    tool: str               # tool name, or "*" for any tool
    condition: str
    action: str             # ALLOW | BLOCK | HUMAN_REVIEW
    reason: str
    priority: int = 0


# backend/agents/policy.py -> backend/agents -> backend -> repo root
DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "policies" / "agent_tools.yaml"
if not DEFAULT_POLICY_PATH.exists():
    for p in Path(__file__).resolve().parents:
        cand = p / "policies" / "agent_tools.yaml"
        if cand.exists():
            DEFAULT_POLICY_PATH = cand
            break


def load_rules(path: Optional[str | Path] = None) -> List[PolicyRule]:
    resolved = Path(path) if path else DEFAULT_POLICY_PATH
    if not resolved.exists():
        return []
    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rules = []
    for raw in data.get("rules", []):
        rules.append(PolicyRule(
            id=raw["id"],
            tool=raw.get("tool", "*"),
            condition=raw.get("condition", "True"),
            action=raw["action"],
            reason=raw.get("reason", ""),
            priority=int(raw.get("priority", 0)),
        ))
    # Highest priority first, so a stricter rule can pre-empt a looser one
    # that would otherwise also match.
    rules.sort(key=lambda r: r.priority, reverse=True)
    return rules


def _build_eval_context(ctx: ToolCallContext, risk: ToolRiskSignal) -> Dict[str, Any]:
    flat: Dict[str, Any] = dict(ctx.args)
    flat.update(risk.raw_context)  # e.g. record_contains_pii, is_external_recipient, contains_pii
    flat.update({
        "role": ctx.role,
        "application": ctx.application,
        "session_risk": ctx.session_risk,
        "risk_score": risk.score,
        **risk.factors,             # authorization, magnitude, sensitivity, session_carryover, ...
    })
    return flat


def _safe_eval(condition: str, context: Dict[str, Any]) -> bool:
    try:
        return bool(eval(condition, {"__builtins__": {}}, context))  # noqa: S307 -- trusted, team-authored YAML only
    except NameError:
        # A condition referencing a field this particular tool call doesn't
        # have (e.g. checking `amount` on a send_email call) should just not
        # match, not crash the whole policy pass.
        return False
    except Exception:
        return False


def evaluate(ctx: ToolCallContext, risk: ToolRiskSignal, rules: Optional[List[PolicyRule]] = None) -> Optional[PolicyRule]:
    rules = rules if rules is not None else load_rules()
    eval_context = _build_eval_context(ctx, risk)
    for rule in rules:
        if rule.tool not in (ctx.tool, "*"):
            continue
        if _safe_eval(rule.condition, eval_context):
            return rule
    return None
