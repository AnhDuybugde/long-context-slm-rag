import unittest

from src.qasper_base_rag.data import build_document_chunks, extract_qa_examples


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


if __name__ == "__main__":
    unittest.main()

