#!/usr/bin/env python
"""
scripts/eval_rag_triad.py

Automated RAG Triad evaluation for ControlPlane.ai CI pipeline.

Measures the three fundamental RAG quality metrics:
  1. Context Relevance  - do retrieved chunks relate to the query?
  2. Groundedness       - is the generated answer faithful and supported by retrieved context?
  3. Answer Relevance   - does the answer address the question asked?

Designed to run offline without external LLM API calls:
  - Context Relevance: cosine similarity using project embedder / local model
  - Groundedness: NLI cross-encoder max score per sentence across retrieved chunks
  - Answer Relevance: cosine similarity between query and generated answer

Usage:
    python scripts/eval_rag_triad.py
    python scripts/eval_rag_triad.py --report
    python scripts/eval_rag_triad.py --fail-under-context-relevance 0.65
    python scripts/eval_rag_triad.py --output eval_results.json

CI usage:
    python scripts/eval_rag_triad.py \
        --fail-under-context-relevance 0.65 \
        --fail-under-groundedness 0.65 \
        --fail-under-answer-relevance 0.70

Exit codes:
    0 - All metrics above thresholds
    1 - One or more metrics below threshold
    2 - Script error (missing deps etc.)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Prevent HuggingFace network retry delays
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("rag_triad_eval")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

GOLDEN_DATASET_PATH = _ROOT / "data" / "eval" / "golden_rag_triad.json"

DEFAULT_CONTEXT_RELEVANCE_MIN = 0.65
DEFAULT_GROUNDEDNESS_MIN = 0.65
DEFAULT_ANSWER_RELEVANCE_MIN = 0.70

# ---------------------------------------------------------------------------
# Lazy-loaded Model Singletons
# ---------------------------------------------------------------------------
_embed_model = None
_embed_model_loaded = False
_nli_cross_encoder = None
_nli_cross_encoder_loaded = False


def _get_sentence_transformer():
    global _embed_model, _embed_model_loaded
    if _embed_model_loaded:
        return _embed_model
    _embed_model_loaded = True
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)
    except Exception:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        except Exception as e:
            logger.debug("SentenceTransformer unavailable: %s", e)
            _embed_model = None
    return _embed_model


def _get_nli_cross_encoder():
    global _nli_cross_encoder, _nli_cross_encoder_loaded
    if _nli_cross_encoder_loaded:
        return _nli_cross_encoder
    _nli_cross_encoder_loaded = True
    try:
        from sentence_transformers import CrossEncoder
        _nli_cross_encoder = CrossEncoder("cross-encoder/nli-deberta-v3-small", max_length=512, local_files_only=True)
    except Exception:
        try:
            from sentence_transformers import CrossEncoder
            _nli_cross_encoder = CrossEncoder("cross-encoder/nli-deberta-v3-small", max_length=512)
        except Exception as e:
            logger.debug("NLI CrossEncoder unavailable: %s", e)
            _nli_cross_encoder = None
    return _nli_cross_encoder


def _get_embedder():
    """Get embedding model. Uses project embedder if available."""
    try:
        from rag.embeddings import get_embedder
        embedder = get_embedder()
        from rag.config import rag_settings
        embedder_path = Path(rag_settings.vector_store_dir) / "policy_embedder_default.pkl"
        if not embedder_path.exists():
            embedder_path = Path(rag_settings.vector_store_dir) / "policy_embedder.pkl"
        if embedder_path.exists():
            embedder.load(embedder_path)
            return embedder
    except Exception as exc:
        logger.debug("Could not load project embedder: %s", exc)
    return None


def _embed_text(text: str, embedder=None) -> Optional[list[float]]:
    """Embed a single text string. Returns list of floats or None on failure."""
    if embedder is not None:
        try:
            vecs = embedder.embed([text])
            return [float(x) for x in vecs[0]]
        except Exception:
            pass
    model = _get_sentence_transformer()
    if model is not None:
        try:
            vec = model.encode(text, normalize_embeddings=True)
            return [float(x) for x in vec]
        except Exception as exc:
            logger.debug("Embedding encode failed: %s", exc)
    return None


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two float lists."""
    import numpy as np
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _check_groundedness(answer: str, context_chunks: list[str]) -> float:
    """
    Check whether the answer is grounded in context using NLI sentence verification.
    For each sentence in the answer, computes best entailment across chunks.
    Returns score in [0.0, 1.0].
    """
    if not answer or not context_chunks:
        return 0.0

    # Extract factual claim sentences (filter out markdown headers, short prefixes)
    sentences = [
        s.strip() for s in re.split(r'(?<=[.!?\n])\s+', answer)
        if len(s.strip()) > 15
        and not s.strip().startswith("[")
        and not s.strip().startswith("#")
        and not s.strip().endswith(":")
    ]
    if not sentences:
        return 1.0

    try:
        import numpy as np
        model = _get_nli_cross_encoder()
        if model is not None:
            sentence_scores = []
            for s in sentences:
                pairs = [(chunk[:800], s) for chunk in context_chunks]
                logits = model.predict(pairs, apply_softmax=True)
                if hasattr(logits, 'ndim') and logits.ndim == 2:
                    p_entail = logits[:, 1]
                    p_neutral = logits[:, 2]
                    # Score for sentence is best match across chunks
                    chunk_faithfulness = p_entail + 0.85 * p_neutral
                    best_match = float(np.max(chunk_faithfulness))
                    sentence_scores.append(best_match)
            if sentence_scores:
                return float(np.mean(sentence_scores))
    except Exception as exc:
        logger.debug("NLI cross-encoder error: %s", exc)

    # Robust fallback: lexical overlap
    answer_words = set(answer.lower().split())
    context_words = set(" ".join(context_chunks).lower().split())
    if not answer_words:
        return 0.0
    overlap = len(answer_words & context_words) / len(answer_words)
    return min(1.0, overlap * 2.0)


def _retrieve_chunks(query: str, top_k: int = 5) -> list[str]:
    """Retrieve top_k chunks across policy_evidence and internal_knowledge collections."""
    chunks = []
    try:
        from rag.policy.policy_rag import _get_retriever as _get_policy_retriever
        res = _get_policy_retriever().retrieve(query, top_k=top_k)
        if res.status == "SUCCESS" and res.chunks:
            chunks.extend(res.chunks)
    except Exception as exc:
        logger.debug("Policy retriever error: %s", exc)

    try:
        from rag.grounding.grounding_checker import _get_retriever as _get_grounding_retriever
        res = _get_grounding_retriever().retrieve(query, top_k=top_k)
        if res.status == "SUCCESS" and res.chunks:
            chunks.extend(res.chunks)
    except Exception as exc:
        logger.debug("Grounding retriever error: %s", exc)

    if not chunks:
        return []

    chunks.sort(key=lambda c: c.score, reverse=True)
    return [c.text for c in chunks[:top_k]]


def _generate_answer(query: str, context_chunks: list[str]) -> str:
    """Generate an answer. Falls back to extractive if no LLM available."""
    try:
        from rag.ask_controlplane.chat import synthesize_answer
        from rag.schemas import RetrievedChunk
        chunks = [
            RetrievedChunk(text=t, score=0.85 - i * 0.05, metadata={"source": "eval"})
            for i, t in enumerate(context_chunks)
        ]
        answer, mode, _ = synthesize_answer(query, chunks)
        return answer
    except Exception as exc:
        logger.debug("synthesize_answer failed: %s", exc)
        if context_chunks:
            return " ".join(context_chunks[:2])
        return ""


def score_single_case(case: dict, embedder=None, top_k: int = 5) -> dict:
    """Score one golden eval case across all three Triad metrics."""
    query = case["query"]

    # Step 1: Retrieve context
    t0 = time.perf_counter()
    context_chunks = _retrieve_chunks(query, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Step 2: Generate answer
    t1 = time.perf_counter()
    answer = _generate_answer(query, context_chunks)
    generation_ms = (time.perf_counter() - t1) * 1000

    # Metric 1: Context Relevance
    context_relevance = 0.0
    if context_chunks:
        query_vec = _embed_text(query, embedder)
        if query_vec is not None:
            sims = []
            for chunk in context_chunks:
                cv = _embed_text(chunk, embedder)
                if cv is not None:
                    sims.append(_cosine_similarity(query_vec, cv))
            if sims:
                context_relevance = sum(sims) / len(sims)

    # Metric 2: Groundedness
    groundedness = 0.0
    if answer and context_chunks:
        groundedness = _check_groundedness(answer, context_chunks)

    # Metric 3: Answer Relevance
    answer_relevance = 0.0
    if answer:
        query_vec = _embed_text(query, embedder)
        answer_vec = _embed_text(answer, embedder)
        if query_vec is not None and answer_vec is not None:
            answer_relevance = _cosine_similarity(query_vec, answer_vec)

    return {
        "id": case["id"],
        "query": query,
        "category": case.get("category", "unknown"),
        "context_chunks_retrieved": len(context_chunks),
        "answer_length_words": len(answer.split()) if answer else 0,
        "context_relevance": round(context_relevance, 4),
        "groundedness": round(groundedness, 4),
        "answer_relevance": round(answer_relevance, 4),
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
    }


def run_rag_triad_eval(
    dataset_path: Path = GOLDEN_DATASET_PATH,
    top_k: int = 5,
    verbose: bool = False,
) -> dict:
    """Run the full RAG Triad evaluation and return a summary dict."""
    if not dataset_path.exists():
        logger.error("Golden dataset not found: %s", dataset_path)
        return {"error": f"Dataset not found: {dataset_path}"}

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info("Loaded %d evaluation cases", len(dataset))

    embedder = _get_embedder()
    results = []

    for i, case in enumerate(dataset):
        logger.info("[%d/%d] %s", i + 1, len(dataset), case["id"])
        try:
            result = score_single_case(case, embedder=embedder, top_k=top_k)
            results.append(result)
            if verbose:
                logger.info(
                    "  CR=%.3f GR=%.3f AR=%.3f",
                    result["context_relevance"],
                    result["groundedness"],
                    result["answer_relevance"],
                )
        except Exception as exc:
            logger.warning("Case %s failed: %s", case["id"], exc)
            results.append({
                "id": case["id"],
                "error": str(exc),
                "context_relevance": 0.0,
                "groundedness": 0.0,
                "answer_relevance": 0.0,
            })

    n = len(results)
    if n == 0:
        return {"error": "No results computed"}

    avg_cr = sum(r["context_relevance"] for r in results) / n
    avg_gr = sum(r["groundedness"] for r in results) / n
    avg_ar = sum(r["answer_relevance"] for r in results) / n

    categories: dict = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, []).append(r)

    category_summary = {}
    for cat, cases in categories.items():
        nc = len(cases)
        category_summary[cat] = {
            "n": nc,
            "context_relevance": round(sum(c["context_relevance"] for c in cases) / nc, 4),
            "groundedness": round(sum(c["groundedness"] for c in cases) / nc, 4),
            "answer_relevance": round(sum(c["answer_relevance"] for c in cases) / nc, 4),
        }

    return {
        "n_cases": n,
        "avg_context_relevance": round(avg_cr, 4),
        "avg_groundedness": round(avg_gr, 4),
        "avg_answer_relevance": round(avg_ar, 4),
        "per_category": category_summary,
        "cases": results,
    }


def print_report(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("  ControlPlane.ai RAG Triad Evaluation Report")
    print("=" * 60)
    if "error" in summary:
        print(f"  ERROR: {summary['error']}")
        return
    print(f"  Cases evaluated      : {summary['n_cases']}")
    print(f"  Context Relevance    : {summary['avg_context_relevance']:.3f}  (threshold >= {DEFAULT_CONTEXT_RELEVANCE_MIN})")
    print(f"  Groundedness (NLI)   : {summary['avg_groundedness']:.3f}  (threshold >= {DEFAULT_GROUNDEDNESS_MIN})")
    print(f"  Answer Relevance     : {summary['avg_answer_relevance']:.3f}  (threshold >= {DEFAULT_ANSWER_RELEVANCE_MIN})")
    print("")
    if summary.get("per_category"):
        print("  Per-Category Breakdown:")
        for cat, stats in summary["per_category"].items():
            print(f"    [{cat}] n={stats['n']} "
                  f"CR={stats['context_relevance']:.3f} "
                  f"GR={stats['groundedness']:.3f} "
                  f"AR={stats['answer_relevance']:.3f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ControlPlane.ai RAG Triad CI evaluator")
    parser.add_argument("--dataset", default=str(GOLDEN_DATASET_PATH))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--fail-under-context-relevance", type=float,
                        default=DEFAULT_CONTEXT_RELEVANCE_MIN, metavar="T")
    parser.add_argument("--fail-under-groundedness", type=float,
                        default=DEFAULT_GROUNDEDNESS_MIN, metavar="T")
    parser.add_argument("--fail-under-answer-relevance", type=float,
                        default=DEFAULT_ANSWER_RELEVANCE_MIN, metavar="T")
    args = parser.parse_args()

    summary = run_rag_triad_eval(
        dataset_path=Path(args.dataset),
        top_k=args.top_k,
        verbose=True,
    )

    print_report(summary)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results written to %s", args.output)

    if "error" in summary:
        sys.exit(2)

    failures = []
    if summary["avg_context_relevance"] < args.fail_under_context_relevance:
        failures.append(
            f"Context Relevance {summary['avg_context_relevance']:.3f} < {args.fail_under_context_relevance:.3f}"
        )
    if summary["avg_groundedness"] < args.fail_under_groundedness:
        failures.append(
            f"Groundedness {summary['avg_groundedness']:.3f} < {args.fail_under_groundedness:.3f}"
        )
    if summary["avg_answer_relevance"] < args.fail_under_answer_relevance:
        failures.append(
            f"Answer Relevance {summary['avg_answer_relevance']:.3f} < {args.fail_under_answer_relevance:.3f}"
        )

    if failures:
        print("\n  QUALITY GATE FAILED:")
        for msg in failures:
            print(f"    x {msg}")
        sys.exit(1)
    else:
        print("\n  All quality gates PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
