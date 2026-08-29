"""Loads plain-text/markdown source documents (Section 7: Markdown, TXT
support) from a directory into Chunk objects, ready for embedding."""

from __future__ import annotations

from pathlib import Path

from rag.chunking import chunk_text
from rag.schemas import Chunk


def load_text_file(path: Path, *, extra_metadata: dict | None = None) -> list[Chunk]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such document: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []  # empty document -- caller decides whether that's an error

    metadata = {
        "source": path.stem,
        "document": path.stem.replace("_", " ").title(),
        "document_type": "regulation",
        "version": "prototype",
        **(extra_metadata or {}),
    }
    return chunk_text(text, source_id=path.stem, metadata=metadata)


def load_directory(directory: Path, *, extra_metadata: dict | None = None) -> list[Chunk]:
    """Load every .txt/.md file in a directory (non-recursive)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            chunks.extend(load_text_file(path, extra_metadata=extra_metadata))
        except Exception as exc:
            # Section 15: one malformed document must not kill ingestion.
            import logging
            logging.getLogger("controlplane.rag").warning("Skipping %s: %s", path, exc)
    return chunks
