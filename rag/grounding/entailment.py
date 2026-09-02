"""Entailment / grounding-support scoring (spec Section 3).

Two implementations behind one interface:

  LexicalEntailmentChecker  — TF-IDF-weighted term overlap + structural
                               heuristics (number mismatch, negation).
                               Zero external deps. Default when
                               RAG_NLI_ENABLED is false.

  NLIEntailmentChecker      — Real Natural Language Inference using a
                               cross-encoder (cross-encoder/nli-deberta-v3-small
                               by default). Returns 0–1 where 1.0 = entailed.
                               Active when RAG_NLI_ENABLED=true. Requires
                               sentence-transformers.
                               Degrades gracefully to lexical checker if the
                               model can't be loaded.

The interface is identical so callers (grounding_checker.py) need zero
changes when switching between implementations.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from rag.schemas import RetrievedChunk

logger = logging.getLogger("controlplane.rag.entailment")

_NUMBER_PATTERN = re.compile(r"\b\d[\d,.]*\b")
_NEGATION_PATTERN = re.compile(r"\b(not|no|never|isn't|aren't|don't|doesn't|cannot|can't)\b", re.I)


class BaseEntailmentChecker(ABC):
    @abstractmethod
    def score_entailment(self, claim: str, evidence_text: str) -> float:
        """Returns 0-1: how well evidence_text supports claim."""
        ...


class LexicalEntailmentChecker(BaseEntailmentChecker):
    """TF-IDF-weighted term overlap between claim and evidence, adjusted by
    two structural checks real NLI models pick up on naturally but naive
    token overlap misses entirely:

      - number mismatch: if the claim states a number, and evidence has
        different numbers with none matching, that's real evidence against
        entailment even with otherwise-high word overlap ("12 days" and
        "20 days" share every other word).
      - negation mismatch: a claim and its evidence sharing every content
        word but disagreeing on negation ("employees must" vs "employees
        must not") is a classic case naive overlap scores as strongly
        supported when it's actually the opposite.
    """

    def __init__(self, vectorizer=None):
        self._vectorizer = vectorizer  # optional pre-fit TfidfVectorizer for consistent IDF weights

    def _term_overlap_score(self, claim: str, evidence_text: str) -> float:
        if self._vectorizer is not None:
            try:
                claim_vec = self._vectorizer.transform([claim])
                evidence_vec = self._vectorizer.transform([evidence_text])
                import numpy as np
                num = (claim_vec.multiply(evidence_vec)).sum()
                denom = (claim_vec.multiply(claim_vec)).sum() ** 0.5
                return float(num / denom) if denom > 0 else 0.0
            except Exception:
                pass  # fall through to the unweighted version below

        claim_words = set(w.lower() for w in claim.split() if len(w) > 2)
        evidence_words = set(w.lower() for w in evidence_text.split() if len(w) > 2)
        if not claim_words:
            return 0.0
        return len(claim_words & evidence_words) / len(claim_words)

    def score_entailment(self, claim: str, evidence_text: str) -> float:
        base = self._term_overlap_score(claim, evidence_text)

        claim_numbers = set(_NUMBER_PATTERN.findall(claim))
        evidence_numbers = set(_NUMBER_PATTERN.findall(evidence_text))
        if claim_numbers and evidence_numbers and not (claim_numbers & evidence_numbers):
            base *= 0.3  # claim asserts a specific number; evidence has numbers, none match

        claim_negated = bool(_NEGATION_PATTERN.search(claim))
        evidence_negated = bool(_NEGATION_PATTERN.search(evidence_text))
        if base > 0.5 and claim_negated != evidence_negated:
            base *= 0.4  # high overlap but disagreeing negation is a red flag, not confirmation

        return round(min(1.0, base), 3)


# ---------------------------------------------------------------------------
# NLI-based checker using cross-encoder/nli-deberta-v3-small
# ---------------------------------------------------------------------------

_nli_model = None
_nli_model_loaded: bool = False


def _get_nli_model():
    """Lazy-load the NLI cross-encoder once; return None on failure."""
    global _nli_model, _nli_model_loaded
    if _nli_model_loaded:
        return _nli_model
    _nli_model_loaded = True
    try:
        from rag.config import rag_settings
        from sentence_transformers import CrossEncoder  # type: ignore
        _nli_model = CrossEncoder(rag_settings.nli_model, max_length=512)
        logger.info("NLI grounding model loaded: %s", rag_settings.nli_model)
    except Exception as exc:
        logger.warning(
            "NLI entailment model unavailable (%s). "
            "Run: pip install sentence-transformers  and set RAG_NLI_ENABLED=true. "
            "Falling back to LexicalEntailmentChecker.",
            exc,
        )
        _nli_model = None
    return _nli_model


class NLIEntailmentChecker(BaseEntailmentChecker):
    """Neural NLI grounding using a cross-encoder.

    The NLI model returns one of three labels:
      ENTAILMENT   — evidence supports the claim        → high score (0.7–1.0)
      NEUTRAL      — evidence neither proves nor disproves → medium score (0.4–0.7)
      CONTRADICTION — evidence contradicts the claim     → low score (0.0–0.3)

    We map this to a 0–1 entailment score for downstream use.
    Falls back to LexicalEntailmentChecker if the model fails.
    """

    def __init__(self):
        self._fallback = LexicalEntailmentChecker()

    def score_entailment(self, claim: str, evidence_text: str) -> float:
        model = _get_nli_model()
        if model is None:
            return self._fallback.score_entailment(claim, evidence_text)
        try:
            # Model id2label: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
            # Split evidence into candidate sentences for precise NLI evaluation
            import re
            raw_sentences = re.split(r"(?<=[.!?\n])\s+", evidence_text)
            candidates = [s.strip() for s in raw_sentences if len(s.strip()) > 15]
            if not candidates:
                candidates = [evidence_text]
            else:
                candidates.append(evidence_text)  # also include full text

            pairs = [(premise, claim) for premise in candidates]
            scores = model.predict(pairs)
            import numpy as np
            probs = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
            entailment_probs = probs[:, 1]  # index 1 = ENTAILMENT
            best_idx = int(np.argmax(entailment_probs))
            entailment_prob = float(entailment_probs[best_idx])
            best_premise = candidates[best_idx]

            # Structural guards: catch specific numeric and negation contradictions
            claim_numbers = set(_NUMBER_PATTERN.findall(claim))
            evidence_numbers = set(_NUMBER_PATTERN.findall(best_premise))
            all_evidence_numbers = set(_NUMBER_PATTERN.findall(evidence_text))
            if claim_numbers and all_evidence_numbers and not (claim_numbers & all_evidence_numbers):
                entailment_prob *= 0.1  # claim asserts a number not anywhere in evidence

            claim_negated = bool(_NEGATION_PATTERN.search(claim))
            evidence_negated = bool(_NEGATION_PATTERN.search(best_premise))
            if entailment_prob > 0.4 and claim_negated != evidence_negated:
                entailment_prob *= 0.2  # negation mismatch

            return round(min(1.0, max(0.0, entailment_prob)), 3)
        except Exception as exc:
            logger.warning("NLI prediction failed (%s), using lexical fallback.", exc)
            return self._fallback.score_entailment(claim, evidence_text)




# ---------------------------------------------------------------------------
# Factory — auto-selects based on RAG_NLI_ENABLED config
# ---------------------------------------------------------------------------

def get_entailment_checker(vectorizer=None) -> BaseEntailmentChecker:
    """Returns NLIEntailmentChecker when RAG_NLI_ENABLED=true (and
    sentence-transformers is installed), otherwise LexicalEntailmentChecker.
    The `vectorizer` arg is passed to the lexical fallback for IDF weighting.
    """
    try:
        from rag.config import rag_settings
        if rag_settings.nli_enabled:
            return NLIEntailmentChecker()
    except Exception:
        pass
    return LexicalEntailmentChecker(vectorizer=vectorizer)

