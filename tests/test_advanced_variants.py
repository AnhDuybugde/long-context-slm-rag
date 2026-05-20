import unittest

import numpy as np

from src.qasper_base_rag.advanced_variants import (
    CrossEncoderReranker,
    RaptorConfig,
    RaptorTreeBuilder,
    SemanticChunker,
    SemanticChunkingConfig,
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

if __name__ == "__main__":
    unittest.main()
