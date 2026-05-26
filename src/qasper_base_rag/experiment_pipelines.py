from __future__ import annotations

from typing import Callable

from .chunking import Chunk
from .advanced_variants import (
    ContextualSemanticRerankerPipeline,
    DenseRerankerPipeline,
    E5QwenFilterGeneratorPipeline,
    GraphRagRaptorWideLateChunkingPipeline,
    GraphRagRaptorSentenceSelectWideLateChunkingPipeline,
    OracleGoldContextPromptAblationPipeline,
    OracleGoldContextPipeline,
    RaptorAgglomerativeAbstractivePipeline,
    RaptorExtractivePipeline,
    RaptorGMMAbstractivePipeline,
    RaptorLeidenAbstractivePipeline,
    SemanticDensePipeline,
    SemanticHybridRerankerPipeline,
    SemanticRaptorLeidenRerankerPipeline,
    SemanticRerankerPipeline,
    SemanticRerankerQwenDirectPipeline,
    SentenceSelectWideLateChunkingPipeline,
    SelfRouteSemanticRerankerPipeline,
    WideLateChunkingSemanticRerankerPipeline,
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

ORACLE_GENERATOR_BOOST_CONFIGS: dict[str, dict] = {
    "oracle_gold_context_flan_base_generator_boost": {
        "prompt_mode": "direct",
        "context_order": "u_tail",
        "tail_reminder": True,
        "num_beams": 4,
    },
}

WIDE_LATE_CHUNKING_CONFIGS: dict[str, dict] = {
    "sem_rerank_minilm_wide_latechunk": {
        "retriever_model": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "sem_rerank_e5_wide_latechunk": {
        "retriever_model": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}

GRAPH_RAPTOR_WIDE_LATE_CHUNKING_CONFIGS: dict[str, dict] = {
    "sem_rerank_minilm_wide_latechunk_graphrag_raptor": {
        "retriever_model": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "sem_rerank_e5_wide_latechunk_graphrag_raptor": {
        "retriever_model": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}

SENTENCE_SELECT_WIDE_LATE_CHUNKING_CONFIGS: dict[str, dict] = {
    "sem_rerank_minilm_wide_latechunk_sentence_select": {
        "retriever_model": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "sem_rerank_e5_wide_latechunk_sentence_select": {
        "retriever_model": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "sem_rerank_minilm_wide_latechunk_high_recall_compress": {
        "retriever_model": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "sem_rerank_e5_wide_latechunk_high_recall_compress": {
        "retriever_model": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}

GRAPH_RAPTOR_SENTENCE_SELECT_WIDE_LATE_CHUNKING_CONFIGS: dict[str, dict] = {
    "sem_rerank_minilm_wide_latechunk_graphrag_raptor_sentence_select": {
        "retriever_model": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "sem_rerank_e5_wide_latechunk_graphrag_raptor_sentence_select": {
        "retriever_model": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "sem_rerank_minilm_wide_latechunk_graphrag_raptor_high_recall_compress": {
        "retriever_model": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "sem_rerank_e5_wide_latechunk_graphrag_raptor_high_recall_compress": {
        "retriever_model": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
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
    if variant == "semantic_chunking_reranker":
        return SemanticRerankerPipeline(
            retriever_model=retriever_model,
            generator_model=generator_model,
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=retrieve_k,
            top_k=top_k,
        )
    if variant == "semantic_chunking_hybrid_reranker":
        return SemanticHybridRerankerPipeline(
            retriever_model=retriever_model,
            generator_model=generator_model,
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=retrieve_k,
            top_k=top_k,
        )
    if variant in WIDE_LATE_CHUNKING_CONFIGS:
        return WideLateChunkingSemanticRerankerPipeline(
            generator_model=generator_model,
            reranker_model=reranker_model,
            min_words=120,
            max_words=420,
            breakpoint_threshold=0.45,
            overlap_sentences=2,
            retrieve_k=30,
            top_k=top_k,
            prompt_mode="direct",
            context_order="u_tail",
            tail_reminder=True,
            max_input_tokens=4096,
            **WIDE_LATE_CHUNKING_CONFIGS[variant],
        )
    if variant in GRAPH_RAPTOR_WIDE_LATE_CHUNKING_CONFIGS:
        return GraphRagRaptorWideLateChunkingPipeline(
            generator_model=generator_model,
            reranker_model=reranker_model,
            min_words=120,
            max_words=420,
            breakpoint_threshold=0.45,
            overlap_sentences=2,
            retrieve_k=30,
            top_k=top_k,
            prompt_mode="direct",
            context_order="u_tail",
            tail_reminder=True,
            max_input_tokens=4096,
            graph_tree_mode="local_tree",
            graph_cluster_backend="leiden",
            graph_fallback_backend="agglomerative",
            graph_max_levels=2,
            graph_branch_k=3,
            graph_parent_top_k=6,
            graph_child_candidate_k=24,
            graph_similarity_threshold=0.70,
            graph_include_parent_context=True,
            graph_summary_mode="extractive_first",
            **GRAPH_RAPTOR_WIDE_LATE_CHUNKING_CONFIGS[variant],
        )
    if variant in SENTENCE_SELECT_WIDE_LATE_CHUNKING_CONFIGS:
        high_recall_compress = variant.endswith("_high_recall_compress")
        return SentenceSelectWideLateChunkingPipeline(
            generator_model=generator_model,
            reranker_model=reranker_model,
            min_words=120,
            max_words=420,
            breakpoint_threshold=0.45,
            overlap_sentences=2,
            retrieve_k=30,
            top_k=top_k,
            prompt_mode="extractive",
            context_order="u_tail",
            tail_reminder=True,
            max_input_tokens=4096,
            sentence_max_sentences=8,
            sentence_window=1,
            sentence_min_query_coverage=0.25,
            sentence_min_best_score=0.20,
            sentence_abstain_on_low_support=True,
            sentence_high_recall=high_recall_compress,
            sentence_high_recall_max_sentences=12,
            sentence_high_recall_complex_max_sentences=16,
            sentence_max_per_context=3,
            **SENTENCE_SELECT_WIDE_LATE_CHUNKING_CONFIGS[variant],
        )
    if variant in GRAPH_RAPTOR_SENTENCE_SELECT_WIDE_LATE_CHUNKING_CONFIGS:
        high_recall_compress = variant.endswith("_high_recall_compress")
        return GraphRagRaptorSentenceSelectWideLateChunkingPipeline(
            generator_model=generator_model,
            reranker_model=reranker_model,
            min_words=120,
            max_words=420,
            breakpoint_threshold=0.45,
            overlap_sentences=2,
            retrieve_k=30,
            top_k=top_k,
            prompt_mode="extractive",
            context_order="u_tail",
            tail_reminder=True,
            max_input_tokens=4096,
            graph_tree_mode="local_tree",
            graph_cluster_backend="leiden",
            graph_fallback_backend="agglomerative",
            graph_max_levels=2,
            graph_branch_k=3,
            graph_parent_top_k=6,
            graph_child_candidate_k=24,
            graph_similarity_threshold=0.70,
            graph_include_parent_context=True,
            graph_summary_mode="extractive_first",
            sentence_max_sentences=8,
            sentence_window=1,
            sentence_min_query_coverage=0.25,
            sentence_min_best_score=0.20,
            sentence_abstain_on_low_support=True,
            sentence_high_recall=high_recall_compress,
            sentence_high_recall_max_sentences=12,
            sentence_high_recall_complex_max_sentences=16,
            sentence_max_per_context=3,
            **GRAPH_RAPTOR_SENTENCE_SELECT_WIDE_LATE_CHUNKING_CONFIGS[variant],
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
    if variant == "raptor_gmm_abstractive":
        return RaptorGMMAbstractivePipeline(
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
    if variant == "raptor_agglomerative_abstractive":
        return RaptorAgglomerativeAbstractivePipeline(
            retriever_model=retriever_model,
            generator_model=generator_model,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
        )
    if variant == "semantic_raptor_leiden_reranker":
        return SemanticRaptorLeidenRerankerPipeline(
            retriever_model=retriever_model,
            generator_model=generator_model,
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=retrieve_k,
            top_k=top_k,
        )
    if variant == "contextual_sem_rerank_minilm_flan_base":
        return ContextualSemanticRerankerPipeline(
            retriever_model="sentence-transformers/all-MiniLM-L6-v2",
            generator_model="google/flan-t5-base",
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=retrieve_k,
            top_k=top_k,
        )
    if variant == "sem_rerank_minilm_qwen15_direct":
        return SemanticRerankerQwenDirectPipeline(
            retriever_model="sentence-transformers/all-MiniLM-L6-v2",
            generator_model="Qwen/Qwen2.5-1.5B-Instruct",
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=retrieve_k,
            top_k=top_k,
        )
    if variant == "sem_rerank_minilm_qwen05_direct":
        return SemanticRerankerQwenDirectPipeline(
            retriever_model="sentence-transformers/all-MiniLM-L6-v2",
            generator_model="Qwen/Qwen2.5-0.5B-Instruct",
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=retrieve_k,
            top_k=top_k,
        )
    if variant == "oracle_gold_context_flan_base":
        return OracleGoldContextPipeline(generator_model="google/flan-t5-base")
    if variant in ORACLE_GENERATOR_BOOST_CONFIGS:
        return OracleGoldContextPromptAblationPipeline(
            generator_model="google/flan-t5-base",
            **ORACLE_GENERATOR_BOOST_CONFIGS[variant],
        )
    if variant == "self_route_minilm_abstain":
        return SelfRouteSemanticRerankerPipeline(
            retriever_model="sentence-transformers/all-MiniLM-L6-v2",
            generator_model=generator_model,
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=retrieve_k,
            top_k=top_k,
        )
    if variant == "self_route_e5_abstain":
        return SelfRouteSemanticRerankerPipeline(
            retriever_model="intfloat/e5-base-v2",
            generator_model=generator_model,
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=retrieve_k,
            top_k=top_k,
            query_prefix="query: ",
            passage_prefix="passage: ",
        )
    if variant == "e5_qwen_filter_flan_base":
        return E5QwenFilterGeneratorPipeline(
            generator_model="google/flan-t5-base",
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=30,
            filter_top_k=max(top_k, 8),
        )
    if variant == "e5_qwen_filter_flan_large":
        return E5QwenFilterGeneratorPipeline(
            generator_model="google/flan-t5-large",
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=30,
            filter_top_k=max(top_k, 8),
        )
    if variant == "e5_qwen_compress_only_flan_large":
        return E5QwenFilterGeneratorPipeline(
            generator_model="google/flan-t5-large",
            filter_mode="compress_only",
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=30,
            filter_top_k=max(top_k, 8),
        )
    if variant == "e5_qwen_soft_route_flan_large":
        return E5QwenFilterGeneratorPipeline(
            generator_model="google/flan-t5-large",
            filter_mode="soft_route",
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=30,
            filter_top_k=max(top_k, 8),
        )
    if variant == "e5_qwen_answer_only":
        return E5QwenFilterGeneratorPipeline(
            generator_model="google/flan-t5-large",
            filter_mode="answer_only",
            answer_with_qwen=True,
            reranker_model=reranker_model,
            max_words=chunk_size,
            retrieve_k=30,
            filter_top_k=max(top_k, 8),
        )
    raise ValueError(f"Unknown variant: {variant}")
