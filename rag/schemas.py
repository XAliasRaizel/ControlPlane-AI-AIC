"""Shared data contracts for the RAG layer.

Mirrors the pattern already established in backend/shared/schemas.py:
one canonical shape per concept, imported everywhere rather than
redefined per-module.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

EvidenceStatus = Literal["SUCCESS", "INSUFFICIENT_EVIDENCE", "RETRIEVAL_ERROR", "MODEL_ERROR", "INVALID_REQUEST"]
ClaimVerdict = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "INSUFFICIENT_EVIDENCE"]


class Chunk(BaseModel):
    """One retrievable unit after loading + chunking, before embedding."""
    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """One chunk as returned by the retriever, with its similarity score."""
    text: str
    score: float  # similarity, 0-1, higher = more relevant
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """What every retriever call returns -- always includes a status, never
    silently returns nothing with no explanation (Section 15 requirement:
    never silently fabricate, always an explicit status)."""
    status: EvidenceStatus
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    query: str
    error: Optional[str] = None


class PolicyEvidence(BaseModel):
    """What the Policy Engine receives and attaches to a decision."""
    status: Literal["SUCCESS", "insufficient_evidence", "RETRIEVAL_ERROR"]
    query: str
    citations: list[RetrievedChunk] = Field(default_factory=list)
    summary: Optional[str] = None  # short, extractive -- not a generative claim


class ClaimCheck(BaseModel):
    """One extracted claim from an LLM response, and its grounding verdict."""
    claim: str
    status: ClaimVerdict
    score: float
    evidence: list[RetrievedChunk] = Field(default_factory=list)


class GroundingReport(BaseModel):
    """The full grounding result for one response -- all claims + rollup."""
    response_id: str
    claims: list[ClaimCheck] = Field(default_factory=list)
    overall_status: ClaimVerdict
    overall_score: float


class AskControlPlaneAnswer(BaseModel):
    """What the Ask ControlPlane chat endpoint returns."""
    answer: str
    citations: list[RetrievedChunk] = Field(default_factory=list)
    status: EvidenceStatus
    confidence: float
    generation_mode: str = "extractive"  # "extractive" | "groq"
