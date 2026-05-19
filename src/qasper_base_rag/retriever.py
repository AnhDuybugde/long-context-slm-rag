from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import Chunk


class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.embeddings is None:
            raise RuntimeError("Call index() before search().")
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.matmul(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[index], float(scores[index])) for index in top_indices]

