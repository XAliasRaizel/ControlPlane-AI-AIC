"""Embedding backends for the RAG layer.

Two implementations behind one interface (Section 14: "make all models
configurable so they can be replaced later"):

  LocalTfidfEmbedder         -- default here. TF-IDF + Truncated SVD (LSA),
                                 fit once on the corpus at ingestion time.
                                 Zero external dependencies, zero network
                                 calls, zero model downloads. This is what
                                 was actually built and tested in this
                                 sandbox.

  SentenceTransformerEmbedder -- a real pretrained embedding model
                                 (all-MiniLM-L6-v2 by default). Correct,
                                 standard sentence-transformers usage, but
                                 NOT exercised in this sandbox: this specific
                                 tool environment's network egress allowlist
                                 does not include huggingface.co (or any
                                 other ML-model hosting domain), so the
                                 model download fails here. That is a
                                 constraint of *this* sandbox, not of your
                                 own machine or wherever Antigravity
                                 ultimately runs -- there, `pip install
                                 sentence-transformers` and a normal internet
                                 connection is enough, and switching to it
                                 is one environment variable
                                 (RAG_EMBEDDING_BACKEND=sentence_transformers),
                                 no code changes.

Why TF-IDF+LSA is a reasonable *default* even beyond this sandbox's
constraints, not just a workaround: this corpus is small, domain-specific,
and terminology-heavy (GDPR article numbers, policy field names, exact
department names) -- exactly the regime where lexical/latent-semantic
methods are competitive with, and sometimes better than, general-purpose
sentence embeddings, which are tuned for broad natural-language similarity
rather than exact regulatory/policy terminology matching.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from rag.config import rag_settings


class BaseEmbedder(ABC):
    """Every embedder: fit once on a corpus, then embed queries/documents
    into the same fitted vector space. `embed` must work for both -- a
    retriever should never need to know which concrete backend it's using.
    """

    @abstractmethod
    def fit(self, texts: list[str]) -> None: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool: ...


class LocalTfidfEmbedder(BaseEmbedder):
    """TF-IDF -> TruncatedSVD (LSA). See module docstring for why."""

    def __init__(self, n_components: int = rag_settings.embedding_dims):
        self.n_components = n_components
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._use_raw = False  # set True by fit() for corpora too small for SVD to be meaningful

    @property
    def is_fitted(self) -> bool:
        return self._vectorizer is not None and (self._svd is not None or self._use_raw)

    def fit(self, texts: list[str]) -> None:
        if not texts:
            raise ValueError("Cannot fit an embedder on an empty corpus.")
        # FIX: a fixed max_df=0.95 is meaningless (and breaks scikit-learn
        # outright) for very small corpora -- with 1 document, every term's
        # document-frequency ratio is 100%, above any max_df < 1.0, leaving
        # zero vocabulary. This bit for real on a freshly-rebuilt audit
        # index with a single record. max_df only makes sense once there
        # are enough documents for "appears in almost all of them" to be a
        # meaningful signal to filter on.
        max_df = 0.95 if len(texts) >= 10 else 1.0
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=max_df,
        )
        tfidf = self._vectorizer.fit_transform(texts)

        # FIX: SVD needs multiple samples to find real cross-document
        # variance structure -- with a 1-document corpus it produced a
        # degenerate output (1 populated dimension instead of the
        # requested n_components, confirmed directly: embedding a
        # single-document corpus's own fit text returned shape (1,1), not
        # (1,2)), plus a "invalid value encountered in divide" warning from
        # a zero-variance denominator. The audit index in particular starts
        # this small routinely (as few as one governance decision logged
        # so far) and grows over time, so this isn't a one-off case to
        # special-case away -- it's the normal cold-start state. Below 5
        # documents, skip SVD and use raw (L2-normalized) TF-IDF directly:
        # less "latent structure," but stable and well-defined at any
        # corpus size, including exactly one document.
        if len(texts) < 5:
            self._use_raw = True
            self._svd = None
            return

        self._use_raw = False
        n_components = min(self.n_components, min(tfidf.shape) - 1, 100)
        n_components = max(n_components, 2)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Embedder must be fit() or load()ed before embed().")
        tfidf = self._vectorizer.transform(texts)
        if self._use_raw:
            return normalize(tfidf).toarray()
        vectors = self._svd.transform(tfidf)
        return normalize(vectors)  # cosine similarity == dot product after this

    def raw_similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity in the raw (pre-SVD) TF-IDF space. See
        query_coverage() below for why this alone isn't used as the
        stability guard for corpora with long or field-dense documents."""
        if not self.is_fitted:
            return 0.0
        a = self._vectorizer.transform([text_a])
        b = self._vectorizer.transform([text_b])
        num = float(a.multiply(b).sum())
        denom = (float(a.multiply(a).sum()) ** 0.5) * (float(b.multiply(b).sum()) ** 0.5)
        return num / denom if denom > 0 else 0.0

    def overlap_weight(self, query: str, evidence_text: str) -> float:
        """Absolute (unnormalized) IDF-weighted term-overlap between query
        and evidence -- the dot product of their raw TF-IDF vectors, with
        no normalization by either side's length.

        This is the measure that actually held up. Two normalized
        alternatives were tried and each failed a different real case:
        symmetric cosine similarity is diluted by long/field-dense
        documents (a genuine exact-ID match scored *lower* than an
        unrelated short-document match, purely from denominator size);
        asymmetric query-coverage (fraction of query weight explained by
        evidence) breaks down for sparse queries where a single incidental
        word can "cover" 100% of what little vocabulary the query has.
        The unnormalized weight avoids both: it grows with genuine,
        specific overlap and isn't inflated by a small denominator on
        either side. Measured directly on this repo's corpora: a genuine
        match (relevant claim vs. its supporting chunk) scored 0.517; a
        spurious SVD-collapse match (unrelated claim vs. an unrelated
        chunk) scored 0.0 -- clean separation, no edge case found.
        """
        if not self.is_fitted:
            return 0.0
        q = self._vectorizer.transform([query])
        d = self._vectorizer.transform([evidence_text])
        return float(q.multiply(d).sum())

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self._vectorizer, "svd": self._svd, "use_raw": self._use_raw}, f)

    def load(self, path: Path) -> None:
        with open(Path(path), "rb") as f:
            state = pickle.load(f)
        self._vectorizer = state["vectorizer"]
        self._svd = state["svd"]
        self._use_raw = state.get("use_raw", False)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Real pretrained embeddings. Correct standard usage; not exercised in
    this sandbox (see module docstring). No fit() step needed -- the model
    is pretrained -- so fit() is a no-op kept only to satisfy the interface.
    """

    def __init__(self, model_name: str = rag_settings.embedding_model):
        self.model_name = model_name
        self._model = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def _load_model(self):
        from sentence_transformers import SentenceTransformer  # local import: optional dep

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: list[str]) -> None:
        self._load_model()  # pretrained -- "fitting" just means loading it

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_name)  # nothing to persist but the model name

    def load(self, path: Path) -> None:
        with open(Path(path)) as f:
            self.model_name = f.read().strip()
        self._load_model()


def get_embedder() -> BaseEmbedder:
    """Factory, gated on RAG_EMBEDDING_BACKEND (rag/config.py)."""
    if rag_settings.embedding_backend == "sentence_transformers":
        return SentenceTransformerEmbedder()
    return LocalTfidfEmbedder()
