"""Converts policies/*.yaml into natural-language documents for Policy RAG.

Programmatic by design (spec Section 7: "make ingestion repeatable") --
this reads the same YAML files backend/policy/loader.py already loads for
enforcement, so the RAG corpus can never drift out of sync with the actual
rules the way a hand-written prose summary would the moment someone edits
a policy file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from rag.schemas import Chunk

_CONDITION_DESCRIBERS = {
    "detector_score_at_least": lambda v: " and ".join(
        f"the {k} detector score is at least {val}" for k, val in (v.items() if isinstance(v, dict) else {})
    ),
    "detector_triggered": lambda v: f"the {v} detector is triggered",
    "all_detectors_triggered": lambda v: "all of these detectors are triggered: " + ", ".join(v),
    "any_detector_triggered": lambda v: "any of these detectors are triggered: " + ", ".join(v),
    "risk_at_least": lambda v: f"the overall risk score is at least {v}",
    "application_in": lambda v: "the application is one of: " + ", ".join(v),
    "data_classification_in": lambda v: "the data classification is one of: " + ", ".join(v),
}


def _describe_condition(when) -> str:
    if isinstance(when, str):
        return when
    if not isinstance(when, dict):
        return "no specific condition (always matches)"
    parts = []
    for key, value in when.items():
        describer = _CONDITION_DESCRIBERS.get(key)
        parts.append(describer(value) if describer else f"{key} = {value}")
    return " and ".join(parts) if parts else "no specific condition (always matches)"


def _rule_to_prose(policy_set: str, rule: dict) -> str:
    condition = _describe_condition(rule.get("when") or rule.get("condition") or {})
    reason = rule.get("reason", rule.get("description", "")).rstrip(".")
    action = rule.get("action", "GOVERN")
    return (
        f"Policy rule '{rule.get('id', 'rule')}' (policy set: {policy_set}, priority {rule.get('priority', 0)}). "
        f"{rule.get('description', '')} "
        f"When {condition}, the resulting action is {action}. "
        f"{('Reason given: ' + reason + '.') if reason else ''}"
    ).strip()


def policy_yaml_to_document(path: Path) -> tuple[str, dict]:
    """Returns (prose_text, metadata) for one policies/*.yaml file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        policy = yaml.safe_load(f) or {}

    policy_set = policy.get("policy_set", path.stem)
    scope = policy.get("scope", {})
    if isinstance(scope, dict):
        apps = scope.get("applications", [])
        depts = scope.get("departments", [])
    elif isinstance(scope, str):
        apps = [scope]
        depts = []
    else:
        apps, depts = [], []

    default_action = policy.get("defaults", {}).get("action", "ALLOW") if isinstance(policy.get("defaults"), dict) else "ALLOW"

    lines = [
        f"Policy document: {policy.get('name', path.stem)} (version {policy.get('version', '1')}, "
        f"policy set '{policy_set}').",
        (
            f"Scope: applies to applications {apps} in departments {depts}."
            if apps or depts
            else "Scope: applies globally, to every application and department not covered by a more specific policy."
        ),
        f"Default action when no rule matches: {default_action}.",
        "",
    ]
    for rule in policy.get("rules", []):
        if isinstance(rule, dict):
            lines.append(_rule_to_prose(policy_set, rule))

    text = "\n".join(lines)
    metadata = {
        "source": path.name,
        "document_type": "internal_policy",
        "domain": policy_set,
        "jurisdiction": "global",
        "policy_id": policy_set,
        "version": str(policy.get("version", "1")),
    }
    return text, metadata


def load_policy_corpus(policies_dir: Path) -> list[Chunk]:
    """Load every policies/*.yaml as one prose document each."""
    policies_dir = Path(policies_dir)
    chunks: list[Chunk] = []
    for path in sorted(policies_dir.glob("*.yaml")):
        try:
            text, metadata = policy_yaml_to_document(path)
        except Exception as exc:
            import logging
            logging.getLogger("controlplane.rag").warning(
                "Skipping malformed policy file %s: %s", path, exc
            )
            continue
        if not text.strip():
            continue
        chunks.append(Chunk(chunk_id=f"policy::{path.stem}", text=text, metadata=metadata))
    return chunks
