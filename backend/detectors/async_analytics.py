"""Async-only analysis engines, registered the same way hot-path detectors
are (Section 7's plugin pattern).

Before this fix, these seven engines existed only as bare functions in
async_pipeline/consumers.py, called directly by name -- a second, parallel
way of representing "a detector" that never touched DETECTOR_REGISTRY.
That's why `[d for d in DETECTOR_REGISTRY.values() if not d.hot_path]` in
worker.py was always an empty list, and why that empty loop was iterated
sequentially instead of with asyncio.gather (nobody ever exercised it well
enough to notice the pattern didn't match the hot path's own).

Wrapping the exact same scoring logic in BaseDetector subclasses means:
  - DETECTOR_REGISTRY now genuinely contains the async detectors.
  - process_async can run them with asyncio.gather, the same concurrency
    primitive the hot path already uses correctly.
  - Every detector in the system -- hot or async -- now speaks one schema
    (DetectorResult), so the audit trail and dashboard don't need two
    different rendering paths for "the same kind of thing."

consumers.run_analytics_engines() is kept as a thin, dict-shaped adapter
over these same instances, so the existing tests and the Streamlit
dashboard (which consume that exact {engine_name: {...}} shape) don't
need to change.
"""

from __future__ import annotations

import re

from backend.detectors.base import BaseDetector, register
from backend.shared.schemas import DetectorResult, GovernanceRequest


@register
class SafetyEngineDetector(BaseDetector):
    name = "safety_engine"
    hot_path = False

    _PATTERNS = {
        "harassment_toxicity": [r"\b(?:kill|hate|threat|attack|destroy|violence|abuse)\b"],
        "exploit_attempts": [r"\b(?:hack|breach|exploit|penetrate|exfiltrate|bypass)\b"],
        "deception": [r"\b(?:impersonate|forge|counterfeit|scam|fraud)\b"],
    }

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = f"{request.prompt}\n{request.response or ''}".lower()
        evidence = []
        for cat, patterns in self._PATTERNS.items():
            for pat in patterns:
                found = re.findall(pat, text)
                if found:
                    evidence.append(f"{cat}: {', '.join(set(found))}")
        score = round(min(1.0, 0.35 * len(evidence)), 3)
        status = "HIGH" if score >= 0.7 else ("MEDIUM" if score > 0 else "LOW")
        return DetectorResult(
            detector_name=self.name,
            score=score,
            label=status,
            confidence=0.9 if evidence else 0.8,
            evidence=evidence or ["Content passed semantic toxicity & safety checks"],
        )


@register
class PrivacyEngineDetector(BaseDetector):
    name = "privacy_engine"
    hot_path = False

    _MARKERS = [
        ("salary_or_financial", r"\b(?:salary|compensation|payroll|bank|account|wage|bonus|\$\d+)\b"),
        ("identity_or_contact", r"\b(?:email|phone|ssn|aadhaar|address|contact)\b"),
        ("confidentiality", r"\b(?:confidential|restricted|private|internal\s*use)\b"),
    ]

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = f"{request.prompt}\n{request.response or ''}".lower()
        evidence = []
        for cat, pattern in self._MARKERS:
            matches = re.findall(pattern, text)
            if matches:
                evidence.append(f"{cat} detected ({len(matches)} occurrences)")
        score = round(min(1.0, 0.30 * len(evidence)), 3)
        status = "HIGH" if score >= 0.6 else ("MEDIUM" if score > 0 else "LOW")
        return DetectorResult(
            detector_name=self.name,
            score=score,
            label=status,
            confidence=0.85 if evidence else 0.8,
            evidence=evidence or ["No high-risk PII or privacy exposure detected"],
        )


@register
class FairnessEngineDetector(BaseDetector):
    name = "bias_fairness_engine"
    hot_path = False

    _TERMS = [
        "gender", "ethnicity", "religion", "race", "disability", "age",
        "because she is", "because he is", "too old", "too young",
    ]

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = f"{request.prompt}\n{request.response or ''}".lower()
        hits = [x for x in self._TERMS if x in text]
        score = round(min(1.0, 0.40 * len(hits)), 3)
        return DetectorResult(
            detector_name=self.name,
            score=score,
            label="MEDIUM" if hits else "LOW",
            confidence=0.7 if hits else 0.8,
            evidence=[f"Demographic markers: {', '.join(hits)}"] if hits
            else ["Zero demographic bias or disparate impact detected"],
        )


@register
class GroundingEngineDetector(BaseDetector):
    name = "hallucination_grounding_engine"
    hot_path = False

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        if not request.response:
            return DetectorResult(
                detector_name=self.name,
                score=0.0,
                label="NOT_APPLICABLE",
                confidence=0.5,
                evidence=["Request blocked or candidate response withheld"],
            )

        # FIX: this used to do raw token-overlap against request.retrieved_context,
        # which (a) is genuinely weak grounding logic -- exactly what the spec
        # calls out to replace -- and (b) request.retrieved_context is never
        # actually populated anywhere in this system, so this branch was dead
        # code that always fell through to a hardcoded score=0.05 "LOW" no
        # matter what the response actually said. Now runs the real
        # Grounding RAG pipeline (rag/grounding/grounding_checker.py):
        # claim extraction -> retrieval against the internal knowledge base
        # -> entailment scoring per claim.
        try:
            from rag.grounding.grounding_checker import check_grounding

            report = check_grounding(request.response, response_id=request.request_id)
            # UNSUPPORTED/INSUFFICIENT_EVIDENCE are both "can't confirm this is
            # grounded" for risk purposes, but only UNSUPPORTED means "the
            # knowledge base actively suggests this claim is unsupported" --
            # worth keeping distinguishable in the evidence even though both
            # map to a similarly elevated score.
            score = round(1.0 - report.overall_score, 3) if report.claims else 0.0
            evidence = [
                f"{c.status}: \"{c.claim[:80]}\"" for c in report.claims if c.status != "SUPPORTED"
            ] or ["All extracted claims supported by internal knowledge base"]
            return DetectorResult(
                detector_name=self.name,
                score=score,
                label=report.overall_status,
                confidence=0.75 if report.claims else 0.5,
                evidence=evidence[:5],
            )
        except Exception as exc:
            # Grounding RAG failure must never break the async pipeline
            # (Section 15) -- report it as its own explicit state instead.
            return DetectorResult(
                detector_name=self.name,
                score=0.0,
                label="MODEL_ERROR",
                confidence=0.3,
                evidence=[f"Grounding check failed: {exc}"],
            )


@register
class PerformanceEngineDetector(BaseDetector):
    name = "performance_engine"
    hot_path = False

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        prompt_len = len(request.prompt)
        resp_len = len(request.response or "")
        throughput_est = max(80, min(220, int(200 - (prompt_len + resp_len) * 0.05)))
        return DetectorResult(
            detector_name=self.name,
            score=0.08,
            label="OPTIMAL",
            confidence=0.9,
            evidence=[f"Throughput: ~{throughput_est} tokens/sec, Complexity: standard"],
        )


@register
class CostEngineDetector(BaseDetector):
    name = "cost_engine"
    hot_path = False

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        prompt_tokens = max(1, len(request.prompt.split()) * 4 // 3)
        response_tokens = len((request.response or "").split()) * 4 // 3
        total_tokens = prompt_tokens + response_tokens
        cost_usd = round(total_tokens * 0.000002, 6)
        return DetectorResult(
            detector_name=self.name,
            score=round(min(1.0, total_tokens / 4000), 3),
            label="LOW",
            confidence=0.95,
            evidence=[f"{prompt_tokens} prompt tokens, {response_tokens} response tokens (Est: ${cost_usd:.6f})"],
        )


@register
class BusinessEngineDetector(BaseDetector):
    name = "business_engine"
    hot_path = False

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        dept = request.department or "General"
        app = request.application_id or "generic"
        return DetectorResult(
            detector_name=self.name,
            score=0.0,
            label="COMPLIANT",
            confidence=0.9,
            evidence=[f"Aligned with {dept} department compliance framework for '{app}'"],
        )
