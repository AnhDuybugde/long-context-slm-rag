import unittest

from src.qasper_base_rag.chunking import chunk_words


class ChunkWordsTest(unittest.TestCase):
    def test_chunk_words_uses_overlap(self):
        chunks = chunk_words("one two three four five six", chunk_size=4, overlap=2)

        self.assertEqual(chunks, ["one two three four", "three four five six"])

    def test_chunk_words_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_words("one two three", chunk_size=3, overlap=3)


if __name__ == "__main__":
    unittest.main()

