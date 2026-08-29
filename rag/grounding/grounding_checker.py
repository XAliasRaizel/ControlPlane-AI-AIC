"""Grounding RAG orchestrator (spec Section 3):

    Generated Response -> Claim Extraction -> for each claim ->
    Retriever -> Entailment Check -> per-claim verdict -> overall report

Ties together claim_extractor, the shared retriever infrastructure
(against the 'internal_knowledge' collection), and entailment.py.
"""

from __future__ import annotations

from pathlib import Path

from rag.config import rag_settings
from rag.embeddings import LocalTfidfEmbedder, get_embedder
from rag.grounding.claim_extractor import extract_claims
from rag.grounding.entailment import get_entailment_checker
from rag.retriever import Retriever
from rag.schemas import ClaimCheck, ClaimVerdict, GroundingReport

_GROUNDING_EMBEDDER_PATH = Path(rag_settings.vector_store_dir) / "grounding_embedder.pkl"

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        embedder = get_embedder()
        if isinstance(embedder, LocalTfidfEmbedder) and _GROUNDING_EMBEDDER_PATH.exists():
            embedder.load(_GROUNDING_EMBEDDER_PATH)
        _retriever = Retriever("internal_knowledge", embedder=embedder)
    return _retriever


def _verdict_for_score(score: float, has_evidence: bool) -> ClaimVerdict:
    if not has_evidence:
        return "INSUFFICIENT_EVIDENCE"
    # Calibrated against measured examples on this repo's real internal_kb
    # corpus: a genuinely well-supported claim scored 0.465; a claim
    # mixing one true and one false sub-fact scored 0.356; a claim
    # contradicting the evidence on a specific number scored 0.153.
    if score >= 0.40:
        return "SUPPORTED"
    if score >= rag_settings.grounding_threshold:
        return "PARTIALLY_SUPPORTED"
    return "UNSUPPORTED"


def check_claim(claim: str, top_k: int = 3) -> ClaimCheck:
    """Section 3's per-claim pipeline. Always returns a ClaimCheck with an
    explicit status -- never guesses when evidence is thin (Section 3's
    "must not guess" requirement)."""
    retriever = _get_retriever()
    result = retriever.retrieve(claim, top_k=top_k, min_score=0.05)

    if result.status != "SUCCESS" or not result.chunks:
        return ClaimCheck(claim=claim, status="INSUFFICIENT_EVIDENCE", score=0.0, evidence=[])

    checker = get_entailment_checker(
        vectorizer=getattr(retriever.embedder, "_vectorizer", None)
    )
    best_score = 0.0
    for chunk in result.chunks:
        score = checker.score_entailment(claim, chunk.text)
        best_score = max(best_score, score)

    verdict = _verdict_for_score(best_score, has_evidence=True)
    return ClaimCheck(claim=claim, status=verdict, score=round(best_score, 3), evidence=result.chunks)


def check_grounding(response_text: str, response_id: str = "response") -> GroundingReport:
    """Full Section 3 pipeline for one response."""
    claims = extract_claims(response_text)

    if not claims:
        return GroundingReport(
            response_id=response_id, claims=[],
            overall_status="INSUFFICIENT_EVIDENCE", overall_score=0.0,
        )

    checks = [check_claim(c) for c in claims]

    # Overall status: the worst individual claim determines it -- a
    # response with one unsupported claim among five supported ones is
    # still a response containing a hallucination, not a mostly-fine one.
    severity_order = {"UNSUPPORTED": 0, "INSUFFICIENT_EVIDENCE": 1, "PARTIALLY_SUPPORTED": 2, "SUPPORTED": 3}
    worst = min(checks, key=lambda c: severity_order[c.status])
    overall_score = sum(c.score for c in checks) / len(checks)

    return GroundingReport(
        response_id=response_id, claims=checks,
        overall_status=worst.status, overall_score=round(overall_score, 3),
    )
