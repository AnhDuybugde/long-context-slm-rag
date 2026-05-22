import unittest

import numpy as np

from src.qasper_base_rag.advanced_variants import (
    AgglomerativeRaptorTreeBuilder,
    CrossEncoderReranker,
    GMMRaptorTreeBuilder,
    RaptorConfig,
    RaptorTreeBuilder,
    SemanticChunker,
    SemanticChunkingConfig,
    SemanticRaptorLeidenRerankerPipeline,
    SemanticHybridRerankerPipeline,
    SemanticRerankerPipeline,
)
from src.qasper_base_rag.chunking import Chunk


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def encode(self, texts, **_kwargs):
        return self.vectors[: len(texts)]


class AdvancedVariantsTest(unittest.TestCase):
    def test_semantic_chunker_breaks_on_large_cosine_distance(self):
        chunker = SemanticChunker(
            SemanticChunkingConfig(min_words=1, max_words=30, breakpoint_threshold=0.4, overlap_sentences=0),
            embedder=FakeEmbedder([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]]),
        )

        chunks = chunker.chunk_text("Alpha method works. Alpha method continues. Beta result differs.")

        self.assertEqual(len(chunks), 2)
        self.assertIn("Alpha method continues.", chunks[0])
        self.assertIn("Beta result differs.", chunks[1])

    def test_cross_encoder_reranker_uses_lexical_fallback_when_disabled(self):
        c1 = Chunk("c1", "d1", "Paper", "intro", "dense retrieval background")
        c2 = Chunk("c2", "d1", "Paper", "method", "keyword reranker evidence")
        reranker = CrossEncoderReranker(model_name=None)

        ranked = reranker.rerank("Which chunk has keyword evidence?", [(c1, 0.1), (c2, 0.1)], top_k=1)

        self.assertEqual(ranked[0][0].chunk_id, "c2")

    def test_semantic_reranker_pipeline_reranks_dense_candidates(self):
        c1 = Chunk("c1", "d1", "Paper", "intro", "dense retrieval background")
        c2 = Chunk("c2", "d1", "Paper", "method", "keyword reranker evidence")

        class FakeRetriever:
            def search(self, question, *, top_k):
                self.question = question
                self.top_k = top_k
                return [(c1, 0.9), (c2, 0.8)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                self.question = question
                self.candidates = candidates
                self.top_k = top_k
                return [(c2, 2.0)]

        class FakeGenerator:
            def answer(self, question, contexts):
                self.question = question
                self.contexts = contexts
                return "reranked answer"

        pipeline = SemanticRerankerPipeline.__new__(SemanticRerankerPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.generator = FakeGenerator()
        pipeline.retrieve_k = 20
        pipeline.top_k = 1

        result = pipeline.answer("Which chunk has keyword evidence?")

        self.assertEqual(result["answer"], "reranked answer")
        self.assertEqual(result["contexts"], [c2])
        self.assertEqual(result["scores"], [2.0])
        self.assertEqual(pipeline.retriever.top_k, 20)
        self.assertEqual(pipeline.reranker.top_k, 1)

    def test_semantic_hybrid_reranker_pipeline_fuses_before_reranking(self):
        c1 = Chunk("c1", "d1", "Paper", "intro", "dense-only candidate")
        c2 = Chunk("c2", "d1", "Paper", "method", "keyword bm25 candidate")

        class FakeRetriever:
            def __init__(self, results):
                self.results = results

            def search(self, question, *, top_k):
                self.question = question
                self.top_k = top_k
                return self.results

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                self.question = question
                self.candidates = candidates
                self.top_k = top_k
                return [(c2, 3.0)]

        class FakeGenerator:
            def answer(self, question, contexts):
                self.question = question
                self.contexts = contexts
                return "hybrid reranked answer"

        pipeline = SemanticHybridRerankerPipeline.__new__(SemanticHybridRerankerPipeline)
        pipeline.dense = FakeRetriever([(c1, 0.9), (c2, 0.2)])
        pipeline.sparse = FakeRetriever([(c2, 2.0)])
        pipeline.reranker = FakeReranker()
        pipeline.generator = FakeGenerator()
        pipeline.retrieve_k = 20
        pipeline.top_k = 1

        result = pipeline.answer("Which chunk has keyword evidence?")

        self.assertEqual(result["answer"], "hybrid reranked answer")
        self.assertEqual(result["contexts"], [c2])
        self.assertEqual(pipeline.dense.top_k, 20)
        self.assertEqual(pipeline.sparse.top_k, 20)
        self.assertEqual(pipeline.reranker.top_k, 1)
        self.assertEqual({chunk.chunk_id for chunk, _score in pipeline.reranker.candidates}, {"c1", "c2"})
        self.assertEqual(result["candidate_retrieval"], "semantic_dense_bm25_rrf")

    def test_raptor_tree_builder_creates_parent_nodes(self):
        leaves = [Chunk(f"c{i}", "d1", "Paper", "s", f"Sentence {i}. More evidence.") for i in range(5)]
        builder = RaptorTreeBuilder(RaptorConfig(group_size=2, max_levels=1), embedder=None)

        parents = builder.build(leaves)

        self.assertGreaterEqual(len(parents), 2)
        self.assertTrue(all("raptor_level_1" == parent.section for parent in parents))

    def test_raptor_tree_builder_uses_injected_summarizer(self):
        class FakeSummarizer:
            def summarize(self, chunks):
                return f"summary of {len(chunks)} chunks"

        leaves = [Chunk(f"c{i}", "d1", "Paper", "s", f"Sentence {i}.") for i in range(2)]
        builder = RaptorTreeBuilder(
            RaptorConfig(group_size=2, max_levels=1),
            embedder=None,
            summarizer=FakeSummarizer(),
        )

        parents = builder.build(leaves)

        self.assertEqual(parents[0].text, "summary of 2 chunks")

    def test_gmm_and_agglomerative_builders_create_parent_nodes(self):
        leaves = [Chunk(f"c{i}", "d1", "Paper", "s", f"Sentence {i}.") for i in range(6)]
        vectors = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]

        gmm_builder = GMMRaptorTreeBuilder(
            RaptorConfig(group_size=2, max_levels=1, random_state=7),
            embedder=FakeEmbedder(vectors),
        )
        agglomerative_builder = AgglomerativeRaptorTreeBuilder(
            RaptorConfig(group_size=2, max_levels=1),
            embedder=FakeEmbedder(vectors),
        )

        self.assertGreaterEqual(len(gmm_builder.build(leaves)), 1)
        self.assertEqual(gmm_builder.last_backend, "umap_gmm_soft")
        self.assertGreaterEqual(len(agglomerative_builder.build(leaves)), 2)
        self.assertEqual(agglomerative_builder.last_backend, "agglomerative_dendrogram_n3_n6_root")

    def test_semantic_raptor_leiden_reranker_reranks_collapsed_candidates(self):
        c1 = Chunk("leaf", "d1", "Paper", "method", "leaf evidence")
        c2 = Chunk("parent", "d1", "Paper", "raptor_level_1", "parent summary evidence")

        class FakeRetriever:
            def search(self, question, *, top_k):
                self.question = question
                self.top_k = top_k
                return [(c1, 0.7), (c2, 0.6)]

        class FakeReranker:
            model_name = "fake-reranker"
            load_error = None

            def rerank(self, question, candidates, *, top_k):
                self.question = question
                self.candidates = candidates
                self.top_k = top_k
                return [(c2, 4.0)]

        class FakeGenerator:
            def answer(self, question, contexts):
                self.question = question
                self.contexts = contexts
                return "raptor reranked answer"

        pipeline = SemanticRaptorLeidenRerankerPipeline.__new__(SemanticRaptorLeidenRerankerPipeline)
        pipeline.retriever = FakeRetriever()
        pipeline.reranker = FakeReranker()
        pipeline.generator = FakeGenerator()
        pipeline.retrieve_k = 20
        pipeline.top_k = 1
        pipeline.parent_count = 3
        pipeline.raptor_backend = "leiden"

        result = pipeline.answer("Which evidence?")

        self.assertEqual(result["answer"], "raptor reranked answer")
        self.assertEqual(result["contexts"], [c2])
        self.assertEqual(result["scores"], [4.0])
        self.assertEqual(result["raptor_parent_count"], 3)
        self.assertEqual(result["raptor_backend"], "leiden")
        self.assertEqual(pipeline.retriever.top_k, 20)
        self.assertEqual(pipeline.reranker.top_k, 1)

if __name__ == "__main__":
    unittest.main()
