"""Ingestion orchestrator: Load -> Clean -> Chunk -> Embed -> Store
(spec Section 7), for the two static corpora (policy/regulatory,
internal knowledge for grounding). The audit corpus is built separately
(rag/ask_controlplane/retrieval.py::rebuild_audit_index) since it's
generated from live data, not static files, and grows continuously.

Run with:
    python -m rag.ingestion.ingest
    python -m rag.ingestion.ingest --tenant acme_corp

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

# Legacy paths (used when tenant_id="default" for backwards compat)
POLICY_EMBEDDER_PATH = Path(rag_settings.vector_store_dir) / "policy_embedder.pkl"
GROUNDING_EMBEDDER_PATH = Path(rag_settings.vector_store_dir) / "grounding_embedder.pkl"


def ingest_policy_corpus(repo_root: Path, tenant_id: str = "default") -> int:
    """Regulatory corpus (GDPR/AI Act/HIPAA prototype docs) + policies/*.yaml
    converted to prose -> one 'policy_evidence' collection. Backs both
    Policy RAG and the policy half of Ask ControlPlane.

    When tenant_id != 'default', the collection is namespaced as
    '{tenant_id}__policy_evidence' and the embedder is saved with a
    tenant-specific filename.
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

    # Tenant-namespaced embedder path
    if tenant_id == "default":
        embedder_path = POLICY_EMBEDDER_PATH
    else:
        embedder_path = Path(rag_settings.vector_store_dir) / f"policy_embedder_{tenant_id}.pkl"
    embedder.save(embedder_path)

    vectors = embedder.embed([c.text for c in all_chunks])
    store = VectorStore("policy_evidence", tenant_id=tenant_id)
    store.reset()
    store.upsert(
        ids=[c.chunk_id for c in all_chunks],
        texts=[c.text for c in all_chunks],
        embeddings=vectors,
        metadatas=[c.metadata for c in all_chunks],
    )
    logger.info(
        "policy_evidence (tenant=%s): %d chunks (%d regulatory, %d policy) indexed, "
        "embedder saved to %s",
        tenant_id, len(all_chunks), len(regulatory_chunks), len(policy_chunks), embedder_path,
    )
    return len(all_chunks)


def ingest_grounding_corpus(tenant_id: str = "default") -> int:
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

    if tenant_id == "default":
        embedder_path = GROUNDING_EMBEDDER_PATH
    else:
        embedder_path = Path(rag_settings.vector_store_dir) / f"grounding_embedder_{tenant_id}.pkl"
    embedder.save(embedder_path)

    vectors = embedder.embed([c.text for c in chunks])
    store = VectorStore("internal_knowledge", tenant_id=tenant_id)
    store.reset()
    store.upsert(
        ids=[c.chunk_id for c in chunks],
        texts=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[c.metadata for c in chunks],
    )
    logger.info(
        "internal_knowledge (tenant=%s): %d chunks indexed, embedder saved to %s",
        tenant_id, len(chunks), embedder_path,
    )
    return len(chunks)


def ingest(repo_root: Path | None = None, tenant_id: str = "default") -> tuple:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    logger.info("Ingesting into %s (tenant=%s)", rag_settings.vector_store_dir, tenant_id)
    n_policy = ingest_policy_corpus(repo_root, tenant_id=tenant_id)
    n_grounding = ingest_grounding_corpus(tenant_id=tenant_id)
    logger.info(
        "Done (tenant=%s). %d policy/regulatory chunks, %d grounding chunks.",
        tenant_id, n_policy, n_grounding,
    )
    return n_policy, n_grounding


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ControlPlane.ai RAG ingestion")
    parser.add_argument(
        "--tenant", default="default",
        help="Tenant ID to ingest for (default: 'default')",
    )
    args = parser.parse_args()
    n_policy, n_grounding = ingest(tenant_id=args.tenant)
    if n_policy == 0 or n_grounding == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
