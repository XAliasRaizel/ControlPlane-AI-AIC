"""Persistent vector store, backed by ChromaDB with a zero-dependency fallback.

Embeddings are always supplied explicitly (never ChromaDB's own default
embedding function -- see rag/embeddings.py for why: that function tries to
download its model at first use, which fails in network-restricted
environments including this sandbox). This module only ever stores and
searches vectors it's given.

If ChromaDB is not installed in the environment, a built-in persistent
NumPy-backed vector store is used seamlessly with identical behavior and
schema contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from rag.config import rag_settings

try:
    import chromadb
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False


class _SimpleVectorStore:
    """Zero-dependency persistent vector store using NumPy & JSON storage."""

    def __init__(self, collection_name: str, persist_dir: Path):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.store_file = self.persist_dir / f"{collection_name}.json"
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.store_file.exists():
            try:
                with open(self.store_file, "r", encoding="utf-8") as f:
                    self._records = json.load(f)
            except Exception:
                self._records = {}
        else:
            self._records = {}

    def _save(self):
        with open(self.store_file, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False)

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings,
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        for doc_id, text, emb, meta in zip(ids, texts, embeddings, metadatas):
            clean_meta = {k: ("" if v is None else v) for k, v in meta.items()} or {"_empty": True}
            self._records[doc_id] = {
                "id": doc_id,
                "text": text,
                "embedding": [float(x) for x in emb],
                "metadata": clean_meta,
            }
        self._save()

    def query(
        self,
        query_embedding,
        top_k: int = rag_settings.top_k,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._records:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1.0

        candidates = []
        for rec in self._records.values():
            meta = rec["metadata"]
            if where:
                match = True
                for k, v in where.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            d_vec = np.array(rec["embedding"], dtype=np.float32)
            d_norm = np.linalg.norm(d_vec)
            if d_norm == 0:
                d_norm = 1.0

            cos_sim = float(np.dot(q_vec, d_vec) / (q_norm * d_norm))
            similarity = max(0.0, min(1.0, cos_sim))
            candidates.append({
                "text": rec["text"],
                "metadata": rec["metadata"],
                "score": similarity,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def count(self) -> int:
        return len(self._records)

    def reset(self) -> None:
        self._records = {}
        if self.store_file.exists():
            try:
                self.store_file.unlink()
            except Exception:
                pass


class VectorStore:
    def __init__(self, collection_name: str, persist_dir: str | None = None):
        self.persist_dir = Path(persist_dir or rag_settings.vector_store_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        if _HAS_CHROMADB:
            try:
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
                self._collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self._fallback = None
            except Exception:
                self._fallback = _SimpleVectorStore(collection_name, self.persist_dir)
        else:
            self._fallback = _SimpleVectorStore(collection_name, self.persist_dir)

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings,
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        if self._fallback is not None:
            self._fallback.upsert(ids, texts, embeddings, metadatas)
            return

        clean_meta = [
            {k: ("" if v is None else v) for k, v in m.items()} or {"_empty": True}
            for m in metadatas
        ]
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=[list(map(float, e)) for e in embeddings],
            metadatas=clean_meta,
        )

    def query(
        self,
        query_embedding,
        top_k: int = rag_settings.top_k,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._fallback is not None:
            return self._fallback.query(query_embedding, top_k=top_k, where=where)

        n = self.count()
        if n == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(map(float, query_embedding))],
            n_results=min(top_k, n),
            where=where or None,
        )
        out = []
        docs = result.get("documents") or [[]]
        metas = result.get("metadatas") or [[]]
        dists = result.get("distances") or [[]]
        for text, meta, dist in zip(docs[0], metas[0], dists[0]):
            similarity = max(0.0, 1.0 - dist)
            out.append({"text": text, "metadata": meta, "score": similarity})
        return out

    def count(self) -> int:
        if self._fallback is not None:
            return self._fallback.count()
        return self._collection.count()

    def reset(self) -> None:
        if self._fallback is not None:
            self._fallback.reset()
            return
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name, metadata={"hnsw:space": "cosine"}
        )
