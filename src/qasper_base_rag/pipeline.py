from __future__ import annotations

from .data import build_document_chunks
from .generator import SmallSeq2SeqGenerator
from .retriever import DenseRetriever


class BaseRAGPipeline:
    def __init__(
        self,
        *,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
        chunk_size: int = 180,
        overlap: int = 40,
        top_k: int = 5,
    ):
        self.retriever = DenseRetriever(retriever_model)
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

    def index_document(self, record: dict) -> None:
        chunks = build_document_chunks(
            record,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        self.retriever.index(chunks)

    def answer(self, question: str) -> dict:
        retrieved = self.retriever.search(question, top_k=self.top_k)
        contexts = [chunk for chunk, _score in retrieved]
        answer = self.generator.answer(question, contexts)
        return {
            "answer": answer,
            "contexts": contexts,
            "scores": [score for _chunk, score in retrieved],
        }

