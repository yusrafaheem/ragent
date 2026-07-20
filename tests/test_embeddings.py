import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import ragent.embeddings as embeddings_module
from ragent.embeddings import HashingEmbedder, get_embedder


class TestHashingEmbedder(unittest.TestCase):
    def test_output_shape(self):
        embedder = HashingEmbedder(dim=64)
        vectors = embedder.embed(["hello world", "foo bar baz"])
        self.assertEqual(vectors.shape, (2, 64))

    def test_vectors_are_l2_normalized(self):
        embedder = HashingEmbedder(dim=64)
        vectors = embedder.embed(["the quick brown fox jumps over the lazy dog"])
        norm = np.linalg.norm(vectors[0])
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_empty_string_does_not_crash_and_is_zero_vector(self):
        embedder = HashingEmbedder(dim=32)
        vectors = embedder.embed([""])
        self.assertEqual(vectors.shape, (1, 32))
        self.assertTrue(np.allclose(vectors[0], 0.0))

    def test_deterministic_across_calls(self):
        embedder = HashingEmbedder(dim=64)
        a = embedder.embed(["reproducibility matters"])
        b = embedder.embed(["reproducibility matters"])
        self.assertTrue(np.array_equal(a, b))

    def test_similar_texts_are_more_similar_than_dissimilar_ones(self):
        embedder = HashingEmbedder(dim=256)
        anchor = embedder.embed(["gradient descent optimizes neural network weights"])[0]
        similar = embedder.embed(["neural network weights are optimized by gradient descent"])[0]
        dissimilar = embedder.embed(["the stock market closed higher today on strong earnings"])[0]

        sim_to_similar = float(anchor @ similar)
        sim_to_dissimilar = float(anchor @ dissimilar)
        self.assertGreater(sim_to_similar, sim_to_dissimilar)


class TestGetEmbedder(unittest.TestCase):
    def test_explicit_hashing_backend(self):
        embedder = get_embedder(backend="hashing", dim=128)
        self.assertIsInstance(embedder, HashingEmbedder)
        self.assertEqual(embedder.dim, 128)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            get_embedder(backend="not-a-real-backend")

    def test_auto_falls_back_to_hashing_when_sentence_transformers_unavailable(self):
        # Force the "real" backend to look unavailable, regardless of
        # whether sentence-transformers actually happens to be installed in
        # the environment running this test, so the fallback branch is
        # exercised deterministically everywhere (including the CI job that
        # *does* install the real ML dependencies).
        with mock.patch.object(
            embeddings_module,
            "SentenceTransformerEmbedder",
            side_effect=ImportError("forced for test"),
        ):
            embedder = get_embedder(backend="auto", dim=64)
        self.assertIsInstance(embedder, HashingEmbedder)


if __name__ == "__main__":
    unittest.main()
