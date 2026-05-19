from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str


def chunk_words(
    text: str,
    *,
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[str]:
    """Split text into fixed-size word chunks.

    This is intentionally simple for the baseline. Later experiments can replace
    it with semantic chunking while keeping the rest of the pipeline comparable.
    """
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks

