import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragent.chunking import chunk_text


class TestChunkText(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text("doc1", ""), [])
        self.assertEqual(chunk_text("doc1", "   "), [])

    def test_short_text_returns_single_chunk(self):
        chunks = chunk_text("doc1", "the quick brown fox", chunk_size=10, overlap=2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "the quick brown fox")
        self.assertEqual(chunks[0].doc_id, "doc1")
        self.assertEqual(chunks[0].chunk_id, "doc1::chunk0")

    def test_long_text_produces_overlapping_chunks(self):
        words = [f"w{i}" for i in range(50)]
        text = " ".join(words)
        chunks = chunk_text("doc1", text, chunk_size=10, overlap=3)

        # stride = chunk_size - overlap = 7, so chunks start at 0, 7, 14, ...
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.text.split()), 10)

        # Consecutive chunks should share the overlap words.
        first_words = chunks[0].text.split()
        second_words = chunks[1].text.split()
        overlap_words = first_words[-3:]
        self.assertEqual(second_words[:3], overlap_words)

    def test_last_chunk_not_dropped_when_shorter_than_chunk_size(self):
        words = [f"w{i}" for i in range(25)]
        text = " ".join(words)
        chunks = chunk_text("doc1", text, chunk_size=10, overlap=2)
        # Every word must appear in at least one chunk.
        covered = set()
        for c in chunks:
            covered.update(c.text.split())
        self.assertEqual(covered, set(words))

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ValueError):
            chunk_text("doc1", "a b c", chunk_size=5, overlap=5)
        with self.assertRaises(ValueError):
            chunk_text("doc1", "a b c", chunk_size=5, overlap=-1)

    def test_invalid_chunk_size_raises(self):
        with self.assertRaises(ValueError):
            chunk_text("doc1", "a b c", chunk_size=0, overlap=0)


if __name__ == "__main__":
    unittest.main()
