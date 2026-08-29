"""Entailment / grounding-support scoring (spec Section 3).

Two implementations behind one interface, same pattern as
rag/embeddings.py and for the same reason: a real NLI/HHEM classifier
needs a model download this sandbox's network allowlist blocks (see
rag/embeddings.py's module docstring for the full explanation). What's
here and tested is a TF-IDF-weighted similarity check PLUS light
structural heuristics -- meaningfully more than raw token overlap (the
thing being replaced), but explicitly not a real NLI model. The interface
is designed so a real HHEM/cross-encoder call is a drop-in replacement
for `score_entailment()` alone -- nothing else in the grounding pipeline
would need to change.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from rag.schemas import RetrievedChunk

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


def get_entailment_checker(vectorizer=None) -> BaseEntailmentChecker:
    return LexicalEntailmentChecker(vectorizer=vectorizer)
