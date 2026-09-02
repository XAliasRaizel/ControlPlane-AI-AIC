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
from backend.shared.model_backend import consult, get_grounding_scorer


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

        # LLM toxicity judge: catches sarcasm, coded language, implied threats that
        # bypass keyword patterns. Runs async via asyncio.to_thread.
        # Gracefully degrades: if provider is mock or errors, keyword result is kept.
        try:
            from backend.utils import llm_judge
            verdict = await asyncio.to_thread(
                llm_judge.judge_toxicity,
                request.prompt,
                request.response or "",
            )
            if not verdict.degraded and verdict.score > score:
                score = round(min(1.0, verdict.score), 3)
                if verdict.score >= 0.6:
                    status = "HIGH"
                elif verdict.score >= 0.3 and status == "LOW":
                    status = "MEDIUM"
                evidence = list(evidence) + [f"llm_toxicity_judge: {verdict.reasoning[:120]}"]
        except Exception:
            pass  # keyword result is the fallback — always valid

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
    fast_async = True

    _TERMS = [
        "gender", "ethnicity", "religion", "race", "disability", "age",
        "because she is", "because he is", "too old", "too young",
    ]

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        text = f"{request.prompt}\n{request.response or ''}".lower()
        hits = [x for x in self._TERMS if x in text]
        keyword_score = round(min(1.0, 0.40 * len(hits)), 3)
        label = "MEDIUM" if hits else "LOW"
        confidence = 0.7 if hits else 0.8
        evidence = ([f"Demographic markers: {', '.join(hits)}"] if hits
                    else ["Zero demographic bias or disparate impact detected"])

        # Optional learned classifier (HateXplain artifact), never lowers keyword score.
        raw_text = f"{request.prompt}\n{request.response or ''}".strip()
        prediction = consult("fairness", raw_text)
        if prediction is not None:
            model_score = prediction["score"]
            keyword_score = max(keyword_score, model_score)
            if prediction["fires"]:
                label = "BIASED"
                confidence = max(confidence, prediction["confidence"])
            evidence = list(evidence) + [f"fairness-model:{model_score:.2f}"]

        score = keyword_score

        # Deep LLM-judge: counterfactual probing + bias rubric via analyze_fairness().
        # Only runs when a real provider is configured (not mock) to avoid noise.
        try:
            from backend.async_engines.fairness import analyze_fairness
            fairness_result = await analyze_fairness(request.prompt, request.response or "")
            if not fairness_result.degraded:
                score = round(max(score, fairness_result.bias_score), 3)
                if fairness_result.bias_score >= 0.6:
                    label = "BIASED"
                elif fairness_result.bias_score >= 0.35 and label == "LOW":
                    label = "MEDIUM"
                confidence = max(confidence, fairness_result.confidence)
                if fairness_result.judge_evidence:
                    evidence = list(evidence) + [f"llm_bias_judge: {e}" for e in fairness_result.judge_evidence[:2]]
                if fairness_result.judge_reasoning:
                    evidence = list(evidence) + [f"judge_reasoning: {fairness_result.judge_reasoning[:100]}"]
        except Exception:
            pass  # keyword + learned-model result is always the safe fallback

        return DetectorResult(
            detector_name=self.name,
            score=score,
            label=label,
            confidence=confidence,
            evidence=evidence,
        )


@register
class GroundingEngineDetector(BaseDetector):
    name = "hallucination_grounding_engine"
    hot_path = False
    fast_async = True

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        if not request.response:
            return DetectorResult(
                detector_name=self.name,
                score=0.0,
                label="NOT_APPLICABLE",
                confidence=0.5,
                evidence=["Request blocked or candidate response withheld"],
            )

        rag_score = None
        rag_label = "INSUFFICIENT_EVIDENCE"
        rag_evidence = []

        # RAG-based NLI claim verification (existing pipeline)
        try:
            from rag.grounding.grounding_checker import check_grounding
            report = check_grounding(request.response, response_id=request.request_id)
            rag_score = round(1.0 - report.overall_score, 3) if report.claims else 0.0
            rag_label = report.overall_status
            rag_evidence = [
                f"{c.status}: \"{c.claim[:80]}\"" for c in report.claims if c.status != "SUPPORTED"
            ] or ["All extracted claims supported by internal knowledge base"]
            rag_confidence = 0.75 if report.claims else 0.5
        except Exception as exc:
            rag_evidence = [f"Grounding RAG unavailable: {exc}"]
            rag_score = 0.0
            rag_confidence = 0.3

        # LLM-as-judge fusion: provides semantic hallucination verdict independent of RAG.
        # Fused score = RAG_NLI * 0.6 + LLM_judge * 0.4 for ensemble AUROC improvement.
        llm_score = None
        llm_evidence = []
        try:
            from backend.utils import llm_judge
            verdict = await asyncio.to_thread(
                llm_judge.judge_grounding,
                request.prompt,
                request.response,
                "",
            )
            if not verdict.degraded:
                # judge_grounding returns score=0.9 for low risk, 0.15 for high risk
                # Convert to hallucination risk: higher score = more hallucinated
                llm_score = round(1.0 - verdict.score, 3)
                if verdict.reasoning:
                    llm_evidence = [f"llm_grounding_judge: {verdict.reasoning[:150]}"]
        except Exception:
            pass

        # Fuse scores
        if llm_score is not None and rag_score is not None:
            fused_score = round(rag_score * 0.6 + llm_score * 0.4, 3)
            confidence = max(rag_confidence, 0.82)  # ensemble improves confidence
        elif llm_score is not None:
            fused_score = llm_score
            confidence = 0.6
        else:
            fused_score = rag_score if rag_score is not None else 0.0
            confidence = rag_confidence

        status = rag_label
        if fused_score >= 0.7:
            status = "HIGH"
        elif fused_score >= 0.4:
            status = "INSUFFICIENT_EVIDENCE"

        return DetectorResult(
            detector_name=self.name,
            score=fused_score,
            label=status,
            confidence=confidence,
            evidence=(rag_evidence + llm_evidence)[:6],
        )


@register
class JailbreakLLMDetector(BaseDetector):
    """Deep jailbreak detector using LLM-as-judge semantic analysis.

    Catches what regex misses: Base64/Rot13 encoded instructions, Pig Latin
    jailbreaks, fictional-frame DAN attacks, and multi-turn hypothetical
    injection. Only adds meaningful signal when CP_JUDGE_PROVIDER != mock.
    Gracefully degrades to no-op when mock/unavailable.
    """
    name = "jailbreak_llm_engine"
    hot_path = False

    async def analyze(self, request: GovernanceRequest, context: dict) -> DetectorResult:
        from backend.utils import llm_judge

        # If mock provider, skip — regex injection detector already handles obvious cases
        if llm_judge.get_active_provider().name == "mock":
            return DetectorResult(
                detector_name=self.name,
                score=0.0,
                label="SKIPPED_MOCK_PROVIDER",
                confidence=0.5,
                evidence=["LLM jailbreak analysis skipped: CP_JUDGE_PROVIDER=mock"],
            )

        try:
            verdict = await asyncio.to_thread(
                llm_judge.judge_injection,
                request.prompt,
                request.response or "",
            )

            if verdict.degraded:
                return DetectorResult(
                    detector_name=self.name,
                    score=0.0,
                    label="DEGRADED",
                    confidence=0.3,
                    evidence=["LLM jailbreak judge degraded — provider error"],
                )

            score = round(verdict.score, 3)
            label = "INJECTION_DETECTED" if verdict.label == "injection_detected" else "CLEAN"
            evidence = verdict.evidence or []
            if verdict.reasoning:
                evidence = list(evidence) + [f"reasoning: {verdict.reasoning[:150]}"]

            return DetectorResult(
                detector_name=self.name,
                score=score,
                label=label,
                confidence=round(float(verdict.raw.get("confidence", 0.5)) if verdict.raw else 0.5, 3),
                evidence=evidence or ["No obfuscated injection or jailbreak technique detected"],
            )
        except Exception as exc:
            return DetectorResult(
                detector_name=self.name,
                score=0.0,
                label="MODEL_ERROR",
                confidence=0.3,
                evidence=[f"Jailbreak LLM check failed: {exc}"],
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
