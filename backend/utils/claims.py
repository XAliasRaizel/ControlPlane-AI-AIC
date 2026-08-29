"""
claims.py -- Lightweight claim decomposition utility.

Used by both the hallucination hot-path heuristic (detectors/hallucination.py)
and the deep grounding engine (async_engines/grounding.py) to break an LLM
response into smaller, independently-checkable units ("claims").

Why this exists
----------------
Every production faithfulness metric we researched (RAGAS's Faithfulness
score, Patronus Lynx, Vectara's HHEM used inside a RAG pipeline) scores
hallucination at the CLAIM level, not the whole-response level:

    faithfulness = (# claims supported by context) / (total # claims)

Checking a whole paragraph against a source document in one shot hides
partial hallucination -- a 200-word answer that is 90% correct but invents
one number will pass a whole-response check yet fail a claim-level one.
This module does the splitting step so the rest of the pipeline can operate
claim-by-claim.

This is intentionally dependency-light (no spaCy/NLTK) so it works inside
the ~50ms hot-path budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DIGIT_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")

_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?"
    r"|\b\d{4}\b)",
    re.IGNORECASE,
)

_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_CLAUSE_SPLIT_RE = re.compile(r",\s+(?:and|but|which|who|because)\s+", re.IGNORECASE)


@dataclass
class Claim:
    text: str
    has_number: bool = False
    has_date: bool = False
    numbers: list = field(default_factory=list)
    named_entities: list = field(default_factory=list)
    span: tuple = (0, 0)

    @property
    def is_checkable(self) -> bool:
        return self.has_number or self.has_date or bool(self.named_entities)


def split_into_claims(text: str, max_claims: int = 24) -> list:
    text = (text or "").strip()
    if not text:
        return []

    sentences = _SENTENCE_SPLIT_RE.split(text)
    raw_claims = []
    for sentence in sentences:
        parts = _CLAUSE_SPLIT_RE.split(sentence)
        raw_claims.extend(p.strip() for p in parts if p.strip())

    claims = []
    cursor = 0
    for raw in raw_claims[:max_claims]:
        start = text.find(raw, cursor)
        start = start if start != -1 else cursor
        end = start + len(raw)
        cursor = end

        numbers = DIGIT_TOKEN_RE.findall(raw)
        claims.append(
            Claim(
                text=raw,
                has_number=bool(numbers),
                has_date=bool(_DATE_RE.search(raw)),
                numbers=numbers,
                named_entities=_PROPER_NOUN_RE.findall(raw),
                span=(start, end),
            )
        )
    return claims


def checkable_claims(text: str, max_claims: int = 24) -> list:
    """Convenience wrapper: only the claims worth spending a grounding check on."""
    return [c for c in split_into_claims(text, max_claims=max_claims) if c.is_checkable]
