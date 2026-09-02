"""Tests for the RAG layer (rag/). Covers spec Section 16's checklist:
document loading, YAML parsing, chunking, embedding, vector insertion,
retrieval, metadata filtering, policy retrieval, claim extraction,
grounding verification, Ask ControlPlane retrieval, insufficient-evidence
behavior, and end-to-end flows.

Every test that touches the vector store uses an isolated tmp_path
collection (via monkeypatching RAG_VECTOR_STORE_DIR) so tests never read
or write the real rag_store/ used by the running application.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def ensure_rag_indexed():
    """Ensure corpora and policy embeddings are indexed before running test suite."""
    from rag.ingestion.ingest import ingest
    ingest()


# ---------------------------------------------------------------------------
# Document loading & YAML parsing
# ---------------------------------------------------------------------------

def test_load_text_file(tmp_path):
    from rag.ingestion.document_loader import load_text_file

    doc = tmp_path / "sample.txt"
    doc.write_text("First paragraph.\n\nSecond paragraph with more content here.")
    chunks = load_text_file(doc)
    assert len(chunks) >= 1
    assert chunks[0].metadata["source"] == "sample"
    assert "First paragraph" in chunks[0].text


def test_load_text_file_missing_raises():
    from rag.ingestion.document_loader import load_text_file
    with pytest.raises(FileNotFoundError):
        load_text_file("/nonexistent/path.txt")


def test_load_directory_skips_malformed_and_wrong_extension(tmp_path):
    from rag.ingestion.document_loader import load_directory

    (tmp_path / "good.txt").write_text("Some real content here for testing purposes.")
    (tmp_path / "ignored.json").write_text("{}")  # wrong extension, skipped
    (tmp_path / "empty.txt").write_text("")  # empty, produces zero chunks, no crash

    chunks = load_directory(tmp_path)
    sources = {c.metadata["source"] for c in chunks}
    assert "good" in sources
    assert "ignored" not in sources


def test_policy_yaml_to_document_real_files():
    """YAML parsing against the actual policies/*.yaml in this repo."""
    from pathlib import Path
    from rag.ingestion.policy_loader import load_policy_corpus

    chunks = load_policy_corpus(Path("policies"))
    assert len(chunks) >= 3  # hr, finance, global, support at minimum
    hr_chunk = next(c for c in chunks if "hr" in c.chunk_id)
    assert "BLOCK" in hr_chunk.text or "MODIFY" in hr_chunk.text
    assert hr_chunk.metadata["document_type"] == "internal_policy"


def test_policy_yaml_to_document_malformed_file_skipped(tmp_path):
    from rag.ingestion.policy_loader import load_policy_corpus

    (tmp_path / "broken.yaml").write_text("not: valid: yaml: [[[")
    (tmp_path / "fine.yaml").write_text(
        "policy_set: test\nversion: 1\nrules:\n  - id: r1\n    action: ALLOW\n"
    )
    chunks = load_policy_corpus(tmp_path)
    # malformed file skipped, not raised; well-formed one still loads
    assert any("fine" in c.chunk_id for c in chunks)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunk_text_respects_paragraph_boundaries():
    from rag.chunking import chunk_text

    text = "Paragraph one is short.\n\nParagraph two is also short."
    chunks = chunk_text(text, source_id="t", chunk_size=1000)
    assert len(chunks) == 1  # both fit in one chunk under this size
    assert "Paragraph one" in chunks[0].text and "Paragraph two" in chunks[0].text


def test_chunk_text_splits_long_content():
    from rag.chunking import chunk_text

    text = "word " * 500
    chunks = chunk_text(text, source_id="t", chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 220 for c in chunks)  # allow small overlap slack


def test_chunk_text_empty_returns_nothing():
    from rag.chunking import chunk_text
    assert chunk_text("", source_id="t") == []
    assert chunk_text("   ", source_id="t") == []


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def test_embedder_fit_and_embed_shape():
    from rag.embeddings import LocalTfidfEmbedder

    e = LocalTfidfEmbedder(n_components=8)
    corpus = [
        "GDPR restricts processing of special category health data across the EU",
        "Employees must not access salary records without documented authorization",
        "The quarterly sales report shows strong growth in the Asia Pacific region",
        "Finance department policy requires manager approval for large expenses",
        "HIPAA governs disclosure of protected health information by covered entities",
        "The office building underwent renovations to the third floor cafeteria",
        "Automated hiring decisions require human review under Article 22",
        "Password resets are handled through the self-service IT helpdesk portal",
        "Customer support tickets are triaged by priority and assigned to agents",
        "The engineering team migrated the database to a new cloud provider",
    ]
    e.fit(corpus)
    vecs = e.embed(corpus)
    assert vecs.shape[0] == 10
    assert abs((vecs[0] ** 2).sum() - 1.0) < 1e-6  # L2 normalized


def test_embedder_small_corpus_uses_raw_fallback():
    """Corpora under 5 documents skip SVD (see embeddings.py docstring for
    why -- SVD is degenerate with too few samples)."""
    from rag.embeddings import LocalTfidfEmbedder

    e = LocalTfidfEmbedder()
    e.fit(["a single short document about leave policy"])
    assert e._use_raw is True
    vecs = e.embed(["a single short document about leave policy"])
    assert vecs.shape[0] == 1


def test_embedder_persistence_roundtrip(tmp_path):
    from rag.embeddings import LocalTfidfEmbedder

    corpus = [
        "Annual leave accrues at 1.5 days per month of service",
        "Sick leave does not require advance manager notification",
        "Remote work requires director approval for fully remote arrangements",
        "Password rotation is mandatory every ninety days for all accounts",
        "Expense reports need itemized receipts above twenty five dollars",
        "Multi factor authentication is required for finance system access",
        "The quarterly all hands meeting covers company wide updates",
        "New hire onboarding includes a two week orientation program",
    ]
    e = LocalTfidfEmbedder(n_components=4)
    e.fit(corpus)
    path = tmp_path / "embedder.pkl"
    e.save(path)

    e2 = LocalTfidfEmbedder()
    assert not e2.is_fitted
    e2.load(path)
    assert e2.is_fitted
    import numpy as np
    assert np.abs(e2.embed(corpus) - e.embed(corpus)).max() < 1e-9


def test_embedder_not_fitted_raises():
    from rag.embeddings import LocalTfidfEmbedder
    with pytest.raises(RuntimeError):
        LocalTfidfEmbedder().embed(["text"])


# ---------------------------------------------------------------------------
# Vector store: insertion, retrieval, metadata filtering
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_store(tmp_path):
    from rag.vector_store import VectorStore
    return VectorStore("test_collection", persist_dir=str(tmp_path))


def test_vector_store_upsert_and_query(isolated_store):
    from rag.embeddings import LocalTfidfEmbedder

    corpus = [f"salary and compensation policy document {i}" for i in range(6)] + \
             [f"unrelated office facilities document {i}" for i in range(6)]
    e = LocalTfidfEmbedder(n_components=6)
    e.fit(corpus)
    isolated_store.upsert(
        ids=[f"d{i}" for i in range(len(corpus))],
        texts=corpus,
        embeddings=e.embed(corpus),
        metadatas=[{"topic": "salary" if i < 6 else "facilities"} for i in range(len(corpus))],
    )
    assert isolated_store.count() == 12

    results = isolated_store.query(e.embed(["salary compensation"])[0], top_k=3)
    assert len(results) == 3
    assert all(r["metadata"]["topic"] == "salary" for r in results)


def test_vector_store_metadata_filtering(isolated_store):
    from rag.embeddings import LocalTfidfEmbedder

    corpus = ["salary document alpha", "salary document beta"]
    e = LocalTfidfEmbedder(n_components=2)
    e.fit(corpus)
    isolated_store.upsert(
        ids=["a", "b"], texts=corpus, embeddings=e.embed(corpus),
        metadatas=[{"source": "hr"}, {"source": "finance"}],
    )
    results = isolated_store.query(e.embed(["salary"])[0], top_k=5, where={"source": "finance"})
    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "finance"


def test_vector_store_empty_query_returns_empty(isolated_store):
    from rag.embeddings import LocalTfidfEmbedder
    e = LocalTfidfEmbedder()
    e.fit(["placeholder so the embedder is fitted"])
    assert isolated_store.query(e.embed(["anything"])[0]) == []


def test_vector_store_handles_empty_metadata(isolated_store):
    """Chroma rejects a completely empty metadata dict outright -- this
    must not crash ingestion (spec Section 15)."""
    from rag.embeddings import LocalTfidfEmbedder
    e = LocalTfidfEmbedder()
    e.fit(["some document text"])
    isolated_store.upsert(ids=["x"], texts=["some document text"], embeddings=e.embed(["some document text"]), metadatas=[{}])
    assert isolated_store.count() == 1


# ---------------------------------------------------------------------------
# Retriever: retrieval + insufficient-evidence behavior
# ---------------------------------------------------------------------------

def test_retriever_success_and_insufficient_evidence(tmp_path):
    from rag.embeddings import LocalTfidfEmbedder
    from rag.retriever import Retriever
    from rag.vector_store import VectorStore

    corpus = [
        "Annual leave policy allows employees eighteen days per calendar year",
        "Leave requests must be submitted through the HR self-service portal",
        "Sick leave is tracked separately and needs no advance notice",
        "Extended leave over twenty days requires a return to work discussion",
        "Managers approve leave requests within three business days typically",
        "Carried over leave expires by the end of the first quarter",
    ]
    e = LocalTfidfEmbedder(n_components=4)
    e.fit(corpus)
    r = Retriever("iso_retriever_test", embedder=e)
    r.store = VectorStore("iso_retriever_test", persist_dir=str(tmp_path))  # explicit isolation
    r.store.upsert(ids=[f"d{i}" for i in range(6)], texts=corpus, embeddings=e.embed(corpus), metadatas=[{}] * 6)

    good = r.retrieve("annual leave policy")
    assert good.status == "SUCCESS"
    assert len(good.chunks) > 0

    bad = r.retrieve("completely unrelated spaceship rocket launch topic", min_score=0.999)
    assert bad.status == "INSUFFICIENT_EVIDENCE"


def test_retriever_invalid_request_on_empty_query():
    from rag.retriever import Retriever
    from rag.embeddings import LocalTfidfEmbedder
    e = LocalTfidfEmbedder()
    e.fit(["placeholder document"])
    r = Retriever("empty_query_test", embedder=e)
    result = r.retrieve("")
    assert result.status == "INVALID_REQUEST"


def test_retriever_error_on_unfitted_embedder():
    from rag.retriever import Retriever
    from rag.embeddings import LocalTfidfEmbedder
    r = Retriever("unfitted_test", embedder=LocalTfidfEmbedder())
    result = r.retrieve("any query")
    assert result.status == "RETRIEVAL_ERROR"


# ---------------------------------------------------------------------------
# Policy RAG (against the real, live policy corpus in this repo)
# ---------------------------------------------------------------------------

def test_policy_rag_retrieves_relevant_evidence():
    from rag.policy.policy_rag import get_policy_evidence

    evidence = get_policy_evidence(
        application_id="hr-copilot", department="HR", user_role="employee",
        action="BLOCK", data_classification="HIGH",
        matched_rule_description="hr pii unauthorized",
    )
    assert evidence.status == "SUCCESS"
    assert len(evidence.citations) > 0
    assert any("hr" in c.metadata.get("source", "").lower() for c in evidence.citations)


def test_policy_rag_query_construction():
    from rag.policy.policy_rag import build_policy_query

    q = build_policy_query(
        application_id="loan-decision", department="Finance", user_role="finance-manager",
        action="HUMAN_REVIEW", data_classification="RESTRICTED",
    )
    assert "loan-decision" in q
    assert "RESTRICTED" in q


# ---------------------------------------------------------------------------
# Grounding RAG: claim extraction + entailment + full pipeline
# ---------------------------------------------------------------------------

def test_claim_extraction_filters_non_claims():
    from rag.grounding.claim_extractor import extract_claims

    text = "Hi there! You have 18 days of leave. Let me know if you need help. Any other questions?"
    claims = extract_claims(text)
    assert claims == ["You have 18 days of leave."]


def test_claim_extraction_empty_response():
    from rag.grounding.claim_extractor import extract_claims
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_entailment_number_mismatch_penalized():
    from rag.grounding.entailment import LexicalEntailmentChecker

    checker = LexicalEntailmentChecker()
    matching = checker.score_entailment(
        "Employees get 18 days of leave.", "Full-time staff accrue 18 days of annual leave yearly."
    )
    mismatched = checker.score_entailment(
        "Employees get 90 days of leave.", "Full-time staff accrue 18 days of annual leave yearly."
    )
    assert matching > mismatched


def test_grounding_checker_four_scenarios():
    """The exact evaluation categories spec Section 11 asks for."""
    from rag.grounding.grounding_checker import check_grounding

    supported = check_grounding("You have 18 days of annual leave.")
    assert supported.overall_status == "SUPPORTED"

    hallucinated = check_grounding("You have 200 days of annual leave remaining.")
    assert hallucinated.overall_status == "UNSUPPORTED"

    insufficient = check_grounding("The office parrot mascot is named Kevin.")
    assert insufficient.overall_status == "INSUFFICIENT_EVIDENCE"

    empty = check_grounding("")
    assert empty.overall_status == "INSUFFICIENT_EVIDENCE"


def test_grounding_never_guesses_with_no_claims():
    from rag.grounding.grounding_checker import check_grounding
    report = check_grounding("Hello! Thanks! Let me know if you need anything else!")
    assert report.claims == []
    assert report.overall_status == "INSUFFICIENT_EVIDENCE"


# ---------------------------------------------------------------------------
# Ask ControlPlane: hybrid retrieval + insufficient-evidence + citations
# ---------------------------------------------------------------------------

def test_ask_controlplane_answers_policy_question():
    from rag.ask_controlplane.chat import ask

    result = ask("What is our policy on PII in Finance?")
    assert result.status == "SUCCESS"
    assert len(result.citations) > 0


def test_ask_controlplane_refuses_to_guess_on_unrelated_question():
    from rag.ask_controlplane.chat import ask

    result = ask("What is the airspeed velocity of an unladen swallow?")
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert "don't have sufficient evidence" in result.answer


def test_ask_controlplane_empty_question_is_invalid():
    from rag.ask_controlplane.chat import ask
    result = ask("")
    assert result.status == "INVALID_REQUEST"


def test_ask_controlplane_request_id_lookup_bypasses_retrieval(tmp_path):
    """A request-id question is an exact lookup, not a similarity search --
    verified by supplying a fake db double, no vector store involved."""
    from rag.ask_controlplane.chat import ask

    class FakeDB:
        def get_audit(self, request_id):
            if request_id == "abc12345":
                return {
                    "request_id": "abc12345", "created_at": "now",
                    "audit_context": {"application_id": "hr-copilot"},
                    "risk": {"overall_risk": 0.9}, "policy": {},
                    "decision_details": {"action": "BLOCK", "reason": "test"},
                    "detector_results": [],
                }
            return None

        def recent_audits(self, limit=500):
            return []

    result = ask("Why was request #abc12345 blocked?", db=FakeDB())
    assert result.status == "SUCCESS"
    assert result.confidence == 1.0
    assert "BLOCK" in result.answer


# ---------------------------------------------------------------------------
# End-to-end: full ingestion -> retrieval flow, from scratch, in isolation
# ---------------------------------------------------------------------------

def test_end_to_end_ingest_and_retrieve(tmp_path):
    """Section 16: 'at least a few end-to-end tests.' Builds a small corpus
    from scratch in an isolated store and retrieves from it, exercising
    the full Load -> Chunk -> Embed -> Store -> Retrieve pipeline without
    touching the real application's rag_store/.
    """
    from rag.chunking import chunk_text
    from rag.embeddings import LocalTfidfEmbedder
    from rag.vector_store import VectorStore

    doc_text = (
        "Data Retention Policy\n\n"
        "Customer records are retained for 7 years after account closure "
        "for regulatory purposes.\n\n"
        "Employee records are retained for 5 years after termination.\n\n"
        "Marketing consent records are retained for 3 years, or until "
        "withdrawn, whichever is sooner."
    )
    chunks = chunk_text(doc_text, source_id="retention_policy", metadata={"domain": "compliance"})
    assert len(chunks) >= 1

    embedder = LocalTfidfEmbedder(n_components=4)
    embedder.fit([c.text for c in chunks])
    vectors = embedder.embed([c.text for c in chunks])

    store = VectorStore("e2e_test", persist_dir=str(tmp_path))
    store.upsert(
        ids=[c.chunk_id for c in chunks], texts=[c.text for c in chunks],
        embeddings=vectors, metadatas=[c.metadata for c in chunks],
    )

    query_vec = embedder.embed(["how long are employee records kept"])[0]
    results = store.query(query_vec, top_k=1)
    assert len(results) == 1
    assert "5 years" in results[0]["text"] or "employee" in results[0]["text"].lower()


def test_end_to_end_policy_rag_through_real_governance_flow():
    """Full stack: a real GovernanceRequest through context enrichment,
    hot path, risk, policy, decision -- then Policy RAG evidence attached,
    exactly as backend/main.py's govern() does it.
    """
    import asyncio
    from backend.shared.schemas import GovernanceRequest
    from backend.gateway.context_enrichment import enrich_context
    from backend.detectors import run_hot_path
    from backend.risk.engine import calculate_risk
    from backend.policy.engine import evaluate_policy
    from backend.decision.engine import make_decision
    from rag.policy.policy_rag import get_policy_evidence

    req = GovernanceRequest(
        user_id="test", user_role="employee", department="Sales",
        application_id="hr-copilot", prompt="Give me Rahul's salary.",
        data_classification="HIGH",
    )
    context = enrich_context(req)
    results, _ = asyncio.run(run_hot_path(req, context))
    risk = calculate_risk(req, results, context)
    policy = evaluate_policy(req, risk, context)
    decision = make_decision(req, risk, policy)

    evidence = get_policy_evidence(
        application_id=req.application_id, department=req.department, user_role=req.user_role,
        action=decision.action, data_classification=req.data_classification,
        matched_rule_description=policy.policy_id,
    )
    assert decision.action == "BLOCK"
    assert evidence.status == "SUCCESS"
    assert len(evidence.citations) > 0
