import unittest

from src.qasper_base_rag.chunking import Chunk
from src.qasper_base_rag.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    token_f1,
)


class MetricsTest(unittest.TestCase):
    def test_token_f1_scores_partial_overlap(self):
        score = token_f1("retrieval augmented generation", "retrieval generation")

        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_context_metrics_use_evidence(self):
        contexts = [
            Chunk("c1", "d1", "Paper", "intro", "RAG retrieves evidence before generation."),
            Chunk("c2", "d1", "Paper", "method", "Unrelated optimization details."),
        ]
        evidence = ["retrieves evidence before generation"]

        self.assertGreater(context_recall(contexts, [], evidence), 0.0)
        self.assertGreater(context_precision(contexts, [], evidence), 0.0)

    def test_faithfulness_rewards_supported_claims(self):
        contexts = [Chunk("c1", "d1", "Paper", "intro", "Qasper contains scientific papers.")]

        score = faithfulness("Qasper contains scientific papers.", contexts)

        self.assertEqual(score, 1.0)

    def test_answer_relevancy_uses_gold_answer_when_available(self):
        score = answer_relevancy(
            "Qasper is a scientific question answering dataset.",
            "What is Qasper?",
            ["scientific question answering dataset"],
        )

        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()

