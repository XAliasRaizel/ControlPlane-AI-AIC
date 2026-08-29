"""Ingestion orchestrator: Load -> Clean -> Chunk -> Embed -> Store
(spec Section 7), for the two static corpora (policy/regulatory,
internal knowledge for grounding). The audit corpus is built separately
(rag/ask_controlplane/retrieval.py::rebuild_audit_index) since it's
generated from live data, not static files, and grows continuously.

Run with:
    python -m rag.ingestion.ingest

Idempotent and safe to re-run -- embedders are re-fit from the current
corpus files each time, and the vector store uses upsert (not insert), so
re-running after editing/adding a document updates it in place rather than
duplicating it. It does NOT rebuild the vector database on every app
startup (Section 7's "avoid rebuilding... every time the application
starts") -- see rag/policy/policy_rag.py and rag/grounding/grounding_checker.py,
which load a persisted, already-fitted index and only fall back to running
this if one doesn't exist yet.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rag.config import rag_settings
from rag.embeddings import get_embedder
from rag.ingestion.document_loader import load_directory
from rag.ingestion.policy_loader import load_policy_corpus
from rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("controlplane.rag.ingest")

POLICY_EMBEDDER_PATH = Path(rag_settings.vector_store_dir) / "policy_embedder.pkl"
GROUNDING_EMBEDDER_PATH = Path(rag_settings.vector_store_dir) / "grounding_embedder.pkl"


def ingest_policy_corpus(repo_root: Path) -> int:
    """Regulatory corpus (GDPR/AI Act/HIPAA prototype docs) + policies/*.yaml
    converted to prose -> one 'policy_evidence' collection. Backs both
    Policy RAG and the policy half of Ask ControlPlane.
    """
    regulatory_chunks = load_directory(
        Path(rag_settings.corpus_dir) / "regulatory",
        extra_metadata={"domain": "regulatory", "jurisdiction": "EU/US"},
    )
    policy_chunks = load_policy_corpus(repo_root / "policies")
    all_chunks = regulatory_chunks + policy_chunks

    if not all_chunks:
        logger.warning("No policy/regulatory documents found -- nothing to ingest.")
        return 0

    embedder = get_embedder()
    embedder.fit([c.text for c in all_chunks])
    embedder.save(POLICY_EMBEDDER_PATH)

    vectors = embedder.embed([c.text for c in all_chunks])
    store = VectorStore("policy_evidence")
    store.reset()
    store.upsert(
        ids=[c.chunk_id for c in all_chunks],
        texts=[c.text for c in all_chunks],
        embeddings=vectors,
        metadatas=[c.metadata for c in all_chunks],
    )
    logger.info(
        "policy_evidence: %d chunks (%d regulatory, %d policy) indexed, embedder saved to %s",
        len(all_chunks), len(regulatory_chunks), len(policy_chunks), POLICY_EMBEDDER_PATH,
    )
    return len(all_chunks)


def ingest_grounding_corpus() -> int:
    """Internal knowledge base (FAQ/handbook) -> 'internal_knowledge'
    collection, used by Grounding RAG to check claims against.
    """
    chunks = load_directory(
        Path(rag_settings.corpus_dir) / "internal_kb",
        extra_metadata={"domain": "internal_knowledge", "document_type": "internal_kb"},
    )
    if not chunks:
        logger.warning("No internal knowledge base documents found -- nothing to ingest.")
        return 0

    embedder = get_embedder()
    embedder.fit([c.text for c in chunks])
    embedder.save(GROUNDING_EMBEDDER_PATH)

    vectors = embedder.embed([c.text for c in chunks])
    store = VectorStore("internal_knowledge")
    store.reset()
    store.upsert(
        ids=[c.chunk_id for c in chunks],
        texts=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[c.metadata for c in chunks],
    )
    logger.info("internal_knowledge: %d chunks indexed, embedder saved to %s", len(chunks), GROUNDING_EMBEDDER_PATH)
    return len(chunks)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    logger.info("Ingesting into %s", rag_settings.vector_store_dir)
    n_policy = ingest_policy_corpus(repo_root)
    n_grounding = ingest_grounding_corpus()
    logger.info("Done. %d policy/regulatory chunks, %d grounding chunks.", n_policy, n_grounding)
    if n_policy == 0 or n_grounding == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
