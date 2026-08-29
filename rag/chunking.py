"""Chunking: turns a loaded document into overlapping, retrievable Chunks.

Simple character-window chunking with overlap, splitting on paragraph/sentence
boundaries where possible rather than mid-word. Deliberately not
token-aware (would need a tokenizer tied to a specific model) -- fine for a
prototype at this corpus size, called out in the Limitations doc.
"""

from __future__ import annotations

import re

from rag.config import rag_settings
from rag.schemas import Chunk


def _split_on_boundaries(text: str) -> list[str]:
    """Prefer splitting on blank lines, then sentence ends, so chunk
    boundaries land somewhere readable instead of mid-sentence."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs if paragraphs else [text]


def chunk_text(
    text: str,
    *,
    source_id: str,
    metadata: dict | None = None,
    chunk_size: int = rag_settings.chunk_size_chars,
    overlap: int = rag_settings.chunk_overlap_chars,
) -> list[Chunk]:
    """Chunk one document's text into overlapping Chunk objects."""
    text = text.strip()
    if not text:
        return []

    metadata = dict(metadata or {})
    paragraphs = _split_on_boundaries(text)

    chunks: list[Chunk] = []
    buffer = ""
    idx = 0

    def flush(buf: str):
        nonlocal idx
        buf = buf.strip()
        if not buf:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{source_id}::{idx}",
                text=buf,
                metadata={**metadata, "chunk_index": idx},
            )
        )
        idx += 1

    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= chunk_size:
            buffer = f"{buffer}\n\n{para}" if buffer else para
            continue

        if len(para) > chunk_size:
            # a single paragraph longer than one chunk: flush what we have,
            # then hard-split the long paragraph itself with overlap.
            flush(buffer)
            buffer = ""
            start = 0
            while start < len(para):
                end = start + chunk_size
                flush(para[start:end])
                start = end - overlap if end - overlap > start else end
            continue

        flush(buffer)
        buffer = buffer[-overlap:] + "\n\n" + para if overlap else para

    flush(buffer)
    return chunks
