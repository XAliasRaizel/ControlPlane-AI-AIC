"""
async_engines/grounding.py -- Deep, non-blocking hallucination/faithfulness
scoring engine.

Runs AFTER the response has already been returned to the caller. Techniques:
  1. Claim-level NLI entailment (Vectara HHEM / RAGAS Faithfulness style)
  2. LLM-as-judge grounding verdict
  3. Self-consistency resampling (SelfCheckGPT-style) for no-context case

None of these three signals is reliable alone. Published benchmarks (BEACON,
arXiv:2606.07528) found standalone methods top out ~0.60 AUROC; ensembles
reach ~0.82. That is the concrete justification for fusing multiple signals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from backend.utils.claims import checkable_claims, Claim, DIGIT_TOKEN_RE
from backend.utils import llm_judge

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_ENTAILMENT_LABEL_INDEX = 1

_nli_model = None
_nli_load_failed = False


def _get_nli_model():
    global _nli_model, _nli_load_failed
    if _nli_model is not None or _nli_load_failed:
        return _nli_model
    try:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(MODEL_NAME)
    except Exception:
        _nli_load_failed = True
        _nli_model = None
    return _nli_model


@dataclass
class ClaimVerdict:
    claim: str
    supported: Optional[bool]
    method: str
    confidence: float = 0.0


@dataclass
class GroundingResult:
    hallucination_score: float
    confidence: float
    faithfulness: Optional[float]
    methods_used: list = field(default_factory=list)
    claim_verdicts: list = field(default_factory=list)
    consistency_score: Optional[float] = None
    judge_reasoning: str = ""
    latency_ms: float = 0.0
    degraded: bool = False


def _lexical_overlap_fallback(claim_text: str, context: str) -> bool:
    claim_words = {w.lower() for w in claim_text.split() if len(w) > 3}
    context_words = {w.lower() for w in context.split()}
    if not claim_words:
        return True
    overlap = len(claim_words & context_words) / len(claim_words)
    return overlap >= 0.5


def _score_claims_with_nli(claims: list, context: str):
    model = _get_nli_model()
    verdicts = []
    used_fallback = model is None
    scores = None

    if model is not None:
        pairs = [[context, c.text] for c in claims]
        try:
            scores = model.predict(pairs)
        except Exception:
            used_fallback = True

    for i, claim in enumerate(claims):
        if scores is not None:
            raw = scores[i]
            try:
                entail_prob = float(raw[_ENTAILMENT_LABEL_INDEX]) if hasattr(raw, "__len__") else float(raw)
            except (TypeError, IndexError):
                entail_prob = float(raw) if isinstance(raw, (int, float)) else 0.5
            verdicts.append(
                ClaimVerdict(claim=claim.text, supported=entail_prob >= 0.5, method="nli",
                             confidence=abs(entail_prob - 0.5) * 2)
            )
        else:
            supported = _lexical_overlap_fallback(claim.text, context)
            verdicts.append(
                ClaimVerdict(claim=claim.text, supported=supported,
                             method="lexical_overlap_fallback", confidence=0.3)
            )
    return verdicts, used_fallback


def _token_overlap(a: str, b: str) -> float:
    a_words = {w.lower() for w in a.split() if len(w) > 3}
    b_words = {w.lower() for w in b.split() if len(w) > 3}
    if not a_words:
        return 0.5
    return len(a_words & b_words) / len(a_words)


def _consistency_from_samples(response: str, samples: list) -> float:
    if not samples:
        return 0.5

    original_claims = checkable_claims(response)
    if not original_claims:
        return sum(_token_overlap(response, s) for s in samples) / len(samples)

    per_sample_scores = []
    for sample in samples:
        sample_lower = sample.lower()
        hits, checks = 0, 0
        for claim in original_claims:
            for number in claim.numbers:
                checks += 1
                hits += 1 if number in DIGIT_TOKEN_RE.findall(sample) else 0
            for entity in claim.named_entities:
                checks += 1
                hits += 1 if entity.lower() in sample_lower else 0
        per_sample_scores.append(hits / checks if checks else _token_overlap(response, sample))
    return sum(per_sample_scores) / len(per_sample_scores)


async def analyze_grounding(prompt: str, response: str, context: str = "") -> GroundingResult:
    t0 = time.perf_counter()
    methods_used = []
    claims = checkable_claims(response)

    claim_verdicts = []
    faithfulness = None
    consistency_score = None
    degraded = False

    if context and claims:
        claim_verdicts, used_fallback = _score_claims_with_nli(claims, context)
        methods_used.append("lexical_overlap_fallback" if used_fallback else "nli_entailment")
        degraded = degraded or used_fallback
        supported_count = sum(1 for v in claim_verdicts if v.supported)
        faithfulness = supported_count / len(claim_verdicts) if claim_verdicts else None

    elif not context:
        if llm_judge.get_active_provider().name == "mock":
            methods_used.append("self_consistency_resampling(skipped_mock_provider)")
            degraded = True
        else:
            samples = llm_judge.resample_for_consistency(prompt, n=3)
            consistency_score = _consistency_from_samples(response, samples)
            methods_used.append("self_consistency_resampling")
            degraded = degraded or (len(samples) == 0)

    judge_verdict = llm_judge.judge_grounding(prompt, response, context)
    methods_used.append(f"llm_judge:{judge_verdict.provider}")
    degraded = degraded or judge_verdict.degraded

    risk_components = []
    if faithfulness is not None:
        risk_components.append(1.0 - faithfulness)
    if consistency_score is not None:
        risk_components.append(1.0 - consistency_score)
    risk_components.append(1.0 - judge_verdict.score)

    hallucination_score = sum(risk_components) / len(risk_components) if risk_components else 0.0
    confidence = 0.9 if len(risk_components) >= 2 and not degraded else (0.6 if not degraded else 0.35)

    return GroundingResult(
        hallucination_score=round(hallucination_score, 4),
        confidence=confidence,
        faithfulness=faithfulness,
        methods_used=methods_used,
        claim_verdicts=claim_verdicts,
        consistency_score=consistency_score,
        judge_reasoning=judge_verdict.reasoning,
        latency_ms=(time.perf_counter() - t0) * 1000,
        degraded=degraded,
    )
