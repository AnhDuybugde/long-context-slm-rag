import json
import tempfile
import unittest
from pathlib import Path

from src.qasper_base_rag.chunking import Chunk
from src.qasper_base_rag.trainer import BaseRAGConfig, BaseRAGTrainer


class FakePipeline:
    def __init__(self):
        self.indexed_doc_ids = []

    def index_document(self, record):
        self.indexed_doc_ids.append(record["id"])

    def answer(self, question):
        chunk = Chunk(
            chunk_id="doc-1::0",
            doc_id="doc-1",
            title="A Paper",
            section="abstract",
            text="The system uses a dense retriever.",
        )
        return {
            "answer": "A dense retriever.",
            "contexts": [chunk],
            "scores": [0.9],
        }


class BaseRAGTrainerTest(unittest.TestCase):
    def test_run_writes_predictions_and_summary(self):
        record = {
            "id": "doc-1",
            "title": "A Paper",
            "abstract": "The system uses a dense retriever.",
            "full_text": {"section_name": [], "paragraphs": []},
            "qas": {
                "question": ["What does the system use?"],
                "question_id": ["q1"],
                "answers": [
                    {
                        "answer": [
                            {
                                "free_form_answer": "A dense retriever.",
                                "evidence": ["The system uses a dense retriever."],
                                "unanswerable": False,
                                "extractive_spans": [],
                                "yes_no": None,
                            }
                        ],
                        "annotation_id": ["a1"],
                        "worker_id": ["w1"],
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = Path(tmpdir) / "predictions.jsonl"
            summary = Path(tmpdir) / "summary.json"
            config = BaseRAGConfig(
                limit=1,
                output_predictions=str(predictions),
                output_summary=str(summary),
            )

            result = BaseRAGTrainer(config, pipeline=FakePipeline()).run([record])

            self.assertEqual(result["metrics"]["examples"], 1)
            self.assertTrue(predictions.exists())
            self.assertTrue(summary.exists())

            prediction_row = json.loads(predictions.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(prediction_row["prediction"], "A dense retriever.")
            self.assertEqual(prediction_row["contexts"][0]["chunk_id"], "doc-1::0")

    def test_none_limit_runs_all_examples(self):
        records = []
        for index in range(2):
            records.append(
                {
                    "id": f"doc-{index}",
                    "title": "A Paper",
                    "abstract": "The system uses a dense retriever.",
                    "full_text": {"section_name": [], "paragraphs": []},
                    "qas": {
                        "question": ["What does the system use?"],
                        "question_id": [f"q{index}"],
                        "answers": [
                            {
                                "answer": [
                                    {
                                        "free_form_answer": "A dense retriever.",
                                        "evidence": ["The system uses a dense retriever."],
                                        "unanswerable": False,
                                        "extractive_spans": [],
                                        "yes_no": None,
                                    }
                                ],
                                "annotation_id": ["a1"],
                                "worker_id": ["w1"],
                            }
                        ],
                    },
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BaseRAGConfig(
                limit=None,
                output_predictions=str(Path(tmpdir) / "predictions.jsonl"),
                output_summary=str(Path(tmpdir) / "summary.json"),
            )

            result = BaseRAGTrainer(config, pipeline=FakePipeline()).run(records)

            self.assertEqual(result["metrics"]["examples"], 2)


if __name__ == "__main__":
    unittest.main()
