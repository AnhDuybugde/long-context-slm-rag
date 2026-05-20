import unittest

from src.qasper_base_rag.data import (
    build_document_chunks,
    document_text,
    document_word_count,
    extract_qa_examples,
    is_long_context_record,
    iter_answer_records,
)


class DataTest(unittest.TestCase):
    def test_extract_qa_examples_reads_answers_and_evidence(self):
        record = {
            "id": "doc-1",
            "title": "A Paper",
            "qas": {
                "question": ["What is used?"],
                "question_id": ["q1"],
                "answers": [
                    [
                        {
                            "answer": {
                                "free_form_answer": "A baseline RAG model.",
                                "evidence": ["The paper uses a baseline RAG model."],
                            }
                        }
                    ]
                ],
            },
        }

        examples = extract_qa_examples(record)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].gold_answers, ["A baseline RAG model."])
        self.assertEqual(examples[0].evidence, ["The paper uses a baseline RAG model."])

    def test_extract_qa_examples_reads_parquet_answer_columns(self):
        record = {
            "id": "doc-1",
            "title": "A Paper",
            "qas": {
                "question": ["What is used?"],
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

        examples = extract_qa_examples(record)

        self.assertEqual(examples[0].gold_answers, ["A dense retriever."])
        self.assertEqual(examples[0].evidence, ["The system uses a dense retriever."])

    def test_iter_answer_records_handles_parquet_column_format(self):
        records = iter_answer_records(
            {
                "answer": [{"free_form_answer": "answer one"}],
                "annotation_id": ["a1"],
                "worker_id": ["w1"],
            }
        )

        self.assertEqual(records[0]["answer"]["free_form_answer"], "answer one")
        self.assertEqual(records[0]["annotation_id"], "a1")

    def test_build_document_chunks_includes_abstract_and_sections(self):
        record = {
            "id": "doc-1",
            "title": "A Paper",
            "abstract": "This is the abstract text.",
            "full_text": {
                "section_name": ["Introduction"],
                "paragraphs": [["This is paragraph one.", "This is paragraph two."]],
            },
        }

        chunks = build_document_chunks(record, chunk_size=4, overlap=1)

        self.assertTrue(any(chunk.section == "abstract" for chunk in chunks))
        self.assertTrue(any(chunk.section == "Introduction" for chunk in chunks))

    def test_document_word_count_supports_long_context_filter(self):
        record = {
            "id": "doc-1",
            "abstract": "one two",
            "full_text": {
                "section_name": ["Intro"],
                "paragraphs": [["three four five", "six seven"]],
            },
        }

        self.assertIn("Intro", document_text(record))
        self.assertEqual(document_word_count(record), 8)
        self.assertTrue(is_long_context_record(record, min_words=8))
        self.assertFalse(is_long_context_record(record, min_words=9))


if __name__ == "__main__":
    unittest.main()
