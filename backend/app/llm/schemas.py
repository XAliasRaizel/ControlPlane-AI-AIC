"""
backend/app/llm/schemas.py

Strict Pydantic v2 output schemas for all internal LLM tasks.
Used after LLM generation to validate structured JSON outputs.
"""
from __future__ import annotations
import json
import re
from typing import List, Literal
from pydantic import BaseModel, Field, model_validator


class GovernanceAnalysis(BaseModel):
    """Structured output for the Advanced Inspector and RLHF Judge."""
    is_safe: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    violated_policies: List[str] = Field(default_factory=list)
    citations: List[int] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    recommendation: Literal["allow", "block", "modify", "human_review"] = "human_review"

    @model_validator(mode="after")
    def check_risk_consistency(self) -> "GovernanceAnalysis":
        if self.risk_score >= 0.7 and self.is_safe:
            object.__setattr__(self, "is_safe", False)
        if self.risk_score < 0.3 and not self.is_safe and not self.violated_policies:
            object.__setattr__(self, "is_safe", True)
        return self

    @classmethod
    def from_llm_json(cls, raw: str) -> "GovernanceAnalysis":
        text = raw.strip()
        if text.startswith("```"):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            text = match.group(0) if match else text
        data = json.loads(text)
        return cls.model_validate(data)

    @classmethod
    def safe_parse(cls, raw: str) -> "GovernanceAnalysis":
        try:
            return cls.from_llm_json(raw)
        except Exception:
            return cls(
                is_safe=False,
                risk_score=1.0,
                violated_policies=[],
                citations=[],
                explanation="Could not parse structured output. Treating as high-risk.",
                recommendation="human_review",
            )


class ChatbotAnswer(BaseModel):
    """Structured output for Ask ControlPlane chatbot."""
    answer: str = Field(min_length=1)
    citations: List[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    insufficient_evidence: bool = False

    @classmethod
    def from_llm_text(cls, text: str, evidence_count: int) -> "ChatbotAnswer":
        cited_raw = [int(m) for m in re.findall(r"\[(\d+)\]", text)]
        cited = sorted(set(c for c in cited_raw if 1 <= c <= max(evidence_count, 1)))
        confidence = round(min(1.0, len(cited) / max(evidence_count, 1) * 1.2), 2) if cited else 0.3
        insufficient = any(p in text.lower() for p in [
            "insufficient evidence", "cannot answer", "no evidence", "not in the evidence"
        ])
        return cls(answer=text.strip(), citations=cited, confidence=confidence,
                   insufficient_evidence=insufficient)


class TokenUsage(BaseModel):
    """Token usage record for a single LLM call."""
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0, default=0)
    model: str
    estimated_cost_usd: float = Field(ge=0.0)

    @model_validator(mode="after")
    def set_total(self) -> "TokenUsage":
        if self.total_tokens == 0:
            object.__setattr__(self, "total_tokens", self.prompt_tokens + self.completion_tokens)
        return self
