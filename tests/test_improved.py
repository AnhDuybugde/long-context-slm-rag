import unittest

from src.qasper_base_rag.chunking import Chunk
from src.qasper_base_rag.improved import (
    BM25Retriever,
    reciprocal_rank_fusion,
    recency_heavy_reorder,
    tokenize,
    u_shaped_reorder,
)


class ImprovedRAGTest(unittest.TestCase):
    def test_tokenize_lowercases_and_keeps_numbers(self):
        self.assertEqual(tokenize("BM25 finds QCVN 08-MT:2015!"), ["bm25", "finds", "qcvn", "08", "mt", "2015"])

    def test_bm25_retriever_matches_keyword(self):
        chunks = [
            Chunk("c1", "d1", "Paper", "intro", "dense semantic retrieval"),
            Chunk("c2", "d1", "Paper", "method", "bm25 sparse keyword retrieval"),
        ]
        retriever = BM25Retriever()
        retriever.index(chunks)

        results = retriever.search("keyword", top_k=1)

        self.assertEqual(results[0][0].chunk_id, "c2")

    def test_rrf_combines_ranked_lists(self):
        c1 = Chunk("c1", "d1", "Paper", "intro", "one")
        c2 = Chunk("c2", "d1", "Paper", "method", "two")

        fused = reciprocal_rank_fusion([[(c1, 0.9), (c2, 0.8)], [(c2, 2.0)]], top_k=2)

        self.assertEqual(fused[0][0].chunk_id, "c2")

    def test_u_shaped_reorder_places_second_best_at_end(self):
        chunks = [Chunk(f"c{i}", "d1", "Paper", "s", str(i)) for i in range(5)]

        reordered = u_shaped_reorder(chunks)

        self.assertEqual(reordered[0].chunk_id, "c0")
        self.assertEqual(reordered[-1].chunk_id, "c1")

    def test_recency_heavy_reorder_places_best_at_end(self):
        chunks = [Chunk(f"c{i}", "d1", "Paper", "s", str(i)) for i in range(4)]

        reordered = recency_heavy_reorder(chunks)

        self.assertEqual(reordered[-1].chunk_id, "c0")


if __name__ == "__main__":
    unittest.main()

