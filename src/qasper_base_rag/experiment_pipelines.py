from __future__ import annotations

from typing import Callable

from .chunking import Chunk
from .advanced_variants import (
    DenseRerankerPipeline,
    RaptorExtractivePipeline,
    RaptorLeidenAbstractivePipeline,
    SemanticDensePipeline,
)
from .data import build_document_chunks
from .generator import SmallSeq2SeqGenerator
from .improved import BM25Retriever, reciprocal_rank_fusion, recency_heavy_reorder, u_shaped_reorder
from .pipeline import BaseRAGPipeline
from .retriever import DenseRetriever


ReorderFn = Callable[[list[Chunk]], list[Chunk]]


def score_order(chunks: list[Chunk]) -> list[Chunk]:
    return chunks


REORDERERS: dict[str, ReorderFn] = {
    "score": score_order,
    "u_shape": u_shaped_reorder,
    "recency_heavy": recency_heavy_reorder,
}


class DenseReorderPipeline(BaseRAGPipeline):
    """Dense retrieval with only prompt-context ordering changed."""

    def __init__(
        self,
        *,
        reorder_mode: str,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
        chunk_size: int = 180,
        overlap: int = 40,
        top_k: int = 5,
    ) -> None:
        super().__init__(
            retriever_model=retriever_model,
            generator_model=generator_model,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
        )
        if reorder_mode not in REORDERERS:
            raise ValueError(f"Unknown reorder mode: {reorder_mode}")
        self.reorder_mode = reorder_mode

    def answer(self, question: str) -> dict:
        retrieved = self.retriever.search(question, top_k=self.top_k)
        score_by_id = {chunk.chunk_id: score for chunk, score in retrieved}
        contexts = REORDERERS[self.reorder_mode]([chunk for chunk, _score in retrieved])
        return {
            "answer": self.generator.answer(question, contexts),
            "contexts": contexts,
            "scores": [score_by_id[chunk.chunk_id] for chunk in contexts],
        }


class BM25OnlyPipeline:
    """Sparse keyword retrieval as an independent retrieval method."""

    def __init__(
        self,
        *,
        generator_model: str = "google/flan-t5-base",
        chunk_size: int = 180,
        overlap: int = 40,
        top_k: int = 5,
    ) -> None:
        self.retriever = BM25Retriever()
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

    def index_document(self, record: dict) -> None:
        chunks = build_document_chunks(record, chunk_size=self.chunk_size, overlap=self.overlap)
        self.retriever.index(chunks)

    def answer(self, question: str) -> dict:
        retrieved = self.retriever.search(question, top_k=self.top_k)
        contexts = [chunk for chunk, _score in retrieved]
        return {
            "answer": self.generator.answer(question, contexts),
            "contexts": contexts,
            "scores": [score for _chunk, score in retrieved],
        }


class HybridRRFPipeline:
    """Dense+BM25 RRF as a separate method, not the default research target."""

    def __init__(
        self,
        *,
        reorder_mode: str = "score",
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
        chunk_size: int = 180,
        overlap: int = 40,
        retrieve_k: int = 20,
        top_k: int = 5,
    ) -> None:
        if reorder_mode not in REORDERERS:
            raise ValueError(f"Unknown reorder mode: {reorder_mode}")
        self.dense = DenseRetriever(retriever_model)
        self.sparse = BM25Retriever()
        self.generator = SmallSeq2SeqGenerator(generator_model)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.retrieve_k = retrieve_k
        self.top_k = top_k
        self.reorder_mode = reorder_mode

    def index_document(self, record: dict) -> None:
        chunks = build_document_chunks(record, chunk_size=self.chunk_size, overlap=self.overlap)
        self.dense.index(chunks)
        self.sparse.index(chunks)

    def answer(self, question: str) -> dict:
        dense_results = self.dense.search(question, top_k=self.retrieve_k)
        sparse_results = self.sparse.search(question, top_k=self.retrieve_k)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], top_k=self.top_k)
        score_by_id = {chunk.chunk_id: score for chunk, score in fused}
        contexts = REORDERERS[self.reorder_mode]([chunk for chunk, _score in fused])
        return {
            "answer": self.generator.answer(question, contexts),
            "contexts": contexts,
            "scores": [score_by_id[chunk.chunk_id] for chunk in contexts],
        }


def build_experiment_pipeline(
    variant: str,
    *,
    retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    generator_model: str = "google/flan-t5-base",
    reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    chunk_size: int = 180,
    overlap: int = 40,
    retrieve_k: int = 20,
    top_k: int = 5,
):
    if variant == "base_dense":
        return BaseRAGPipeline(
            retriever_model=retriever_model,
            generator_model=generator_model,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
        )
    if variant == "bm25_only":
        return BM25OnlyPipeline(
            generator_model=generator_model,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
        )
    if variant == "dense_u_shape":
        return DenseReorderPipeline(
            reorder_mode="u_shape",
            retriever_model=retriever_model,
            generator_model=generator_model,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
        )
    if variant == "dense_recency_heavy":
        return DenseReorderPipeline(
            reorder_mode="recency_heavy",
            retriever_model=retriever_model,
            generator_model=generator_model,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
        )
    if variant == "hybrid_rrf":
        return HybridRRFPipeline(
            reorder_mode="score",
            retriever_model=retriever_model,
            generator_model=generator_model,
            chunk_size=chunk_size,
            overlap=overlap,
            retrieve_k=retrieve_k,
            top_k=top_k,
        )
    if variant == "semantic_chunking_dense":
        return SemanticDensePipeline(
            retriever_model=retriever_model,
            generator_model=generator_model,
            max_words=chunk_size,
            top_k=top_k,
        )
    if variant == "dense_reranker":
        return DenseRerankerPipeline(
            retriever_model=retriever_model,
            generator_model=generator_model,
            reranker_model=reranker_model,
            chunk_size=chunk_size,
            overlap=overlap,
            retrieve_k=retrieve_k,
            top_k=top_k,
        )
    if variant == "raptor_extractive":
        return RaptorExtractivePipeline(
            retriever_model=retriever_model,
            generator_model=generator_model,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
        )
    if variant == "raptor_leiden_abstractive":
        return RaptorLeidenAbstractivePipeline(
            retriever_model=retriever_model,
            generator_model=generator_model,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
        )
    raise ValueError(f"Unknown variant: {variant}")
