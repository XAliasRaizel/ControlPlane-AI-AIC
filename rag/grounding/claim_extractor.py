"""Extracts candidate factual claims from an LLM response (spec Section 3).

Sentence-split + filter, rather than an LLM-based claim extractor: no
generation step needed for something this structural, and it keeps claim
extraction fast enough to run in the async grounding path without adding
another model dependency. Filters out greetings, questions, and
meta-commentary that aren't actually checkable factual assertions.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_NON_CLAIM_PATTERNS = [
    re.compile(r"^(hi|hello|hey|thanks|thank you|sure|okay|ok)\b", re.I),
    re.compile(r"\?\s*$"),  # questions aren't claims to verify
    re.compile(r"^(let me know|feel free|please|i can|i'll|i will)\b", re.I),
]

_MIN_CLAIM_WORDS = 4


def extract_claims(response_text: str) -> list[str]:
    """Splits a response into sentences and keeps the ones that look like
    checkable factual statements."""
    if not response_text or not response_text.strip():
        return []

    sentences = _SENTENCE_SPLIT.split(response_text.strip())
    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence.split()) < _MIN_CLAIM_WORDS:
            continue
        if any(pattern.search(sentence) for pattern in _NON_CLAIM_PATTERNS):
            continue
        claims.append(sentence)
    return claims
