"""
backend/app/llm/prompts.py

Task-specific prompt builders for the shared LLMClient.
Both features call the same LLMClient -- this file is where
"which task" lives, not a second client implementation.

v2: Registry-backed builders delegate to versioned Jinja2 templates in
    prompts/ directory. Existing callers are unaffected (zero breaking changes).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional


def build_chatbot_system_prompt() -> str:
    """System prompt for Governance Chatbot / Ask ControlPlane (v1 inline fallback)."""
    return (
        "You are ControlPlane.ai's governance assistant. Answer the user's question "
        "using ONLY the evidence provided to you inside the <evidence> block. Cite "
        "the evidence you use with [N] markers matching the evidence numbering. "
        "Never invent policy names, request IDs, approval counts, regulation names, "
        "or any other fact not present in the evidence. If the evidence doesn't "
        "fully answer the question, say so explicitly rather than filling the gap "
        "yourself. Be concise and cite specific policy names or regulatory articles "
        "where relevant."
    )


def build_inspector_system_prompt() -> str:
    """System prompt for Advanced Inspector -- returns structured JSON."""
    return (
        "You are ControlPlane's governance inspector. Analyze the supplied request "
        "using ONLY the evidence provided inside the <evidence> block. Respond with "
        "a JSON object with EXACTLY these keys: "
        "applicable_policy (string or null), "
        "evidence_refs (list of integers -- only evidence numbers you actually used), "
        "detected_risk (string: 'low', 'medium', or 'high'), "
        "reason (string: explain why), "
        "required_controls (list of strings), "
        "recommendation (string: 'allow', 'block', 'modify', or 'human_review'). "
        "Never invent evidence. If nothing in the evidence applies, use null/empty "
        "values rather than guessing. Respond with ONLY the JSON object, no markdown."
    )


@dataclass
class InspectionResult:
    """Structured result from the Advanced Inspector."""
    applicable_policy: Optional[str]
    evidence_refs: List[int] = field(default_factory=list)
    detected_risk: str = ""
    reason: str = ""
    required_controls: List[str] = field(default_factory=list)
    recommendation: str = ""
    generation_mode: str = "extractive"
    citation_check: Optional[dict] = None
    raw_text: Optional[str] = None  # stored for audit logging


def parse_inspection_result(
    raw_json: str,
    generation_mode: str,
    citation_check: Optional[dict],
) -> InspectionResult:
    """Parse LLM JSON output into InspectionResult.

    Fails honestly on malformed output: detected_risk="unknown", reason
    explains the parse failure. A result that silently looks fine is worse
    than one that visibly says it could not be parsed.
    """
    try:
        # Strip markdown fences if the model wrapped the JSON
        text = raw_json.strip()
        if text.startswith("```"):
            text = text.strip("`")
            start = text.find("{")
            if start != -1:
                text = text[start:]
        data = json.loads(text)
        return InspectionResult(
            applicable_policy=data.get("applicable_policy"),
            evidence_refs=list(data.get("evidence_refs") or []),
            detected_risk=str(data.get("detected_risk", "")),
            reason=str(data.get("reason", "")),
            required_controls=list(data.get("required_controls") or []),
            recommendation=str(data.get("recommendation", "")),
            generation_mode=generation_mode,
            citation_check=citation_check,
            raw_text=raw_json,
        )
    except (json.JSONDecodeError, AttributeError, TypeError):
        return InspectionResult(
            applicable_policy=None,
            detected_risk="unknown",
            reason=(
                "Could not parse a structured result from the model output. "
                "Raw output has been logged for review; treat this request as "
                "unresolved, not as low-risk."
            ),
            generation_mode=generation_mode,
            citation_check=citation_check,
            raw_text=raw_json,
        )


# ---------------------------------------------------------------------------
# Registry-backed prompt builders (v2 — delegates to versioned Jinja2 files)
# ---------------------------------------------------------------------------

def build_chatbot_system_prompt_v2(department: str = "", max_tokens: int = 400) -> str:
    """Return the v2 Jinja2-rendered chatbot prompt with optional department context."""
    try:
        from .prompt_registry import get_registry
        return get_registry().render(
            "ask_controlplane", version="v2",
            department=department, max_tokens=max_tokens,
        )
    except Exception:
        return build_chatbot_system_prompt()  # safe fallback to inline v1


def build_rlhf_judge_prompt() -> str:
    """Return the RLHF judge system prompt from the versioned registry."""
    try:
        from .prompt_registry import get_registry
        return get_registry().render("rlhf_judge", version="v1")
    except Exception:
        return (
            "You are a governance quality judge. Evaluate the following AI decision "
            "against the provided policy evidence and return a JSON analysis."
        )


def build_grounding_extractor_prompt() -> str:
    """Return the grounding extractor prompt from the versioned registry."""
    try:
        from .prompt_registry import get_registry
        return get_registry().render("grounding_extractor", version="v1")
    except Exception:
        return "You are a grounding verification assistant. Check if claims are supported by evidence."
