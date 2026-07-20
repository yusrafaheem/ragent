"""
Exercises the real (non-fallback) embedding and vector-search backends:
sentence-transformers and faiss-cpu.

These are optional extras (see pyproject.toml's [project.optional-dependencies])
and are not installed in the base dev environment, so every test here is
skipped with a visible reason when its dependency is absent, rather than
failing. CI runs this file twice: once in the standard `test` job (where
everything here is skipped, proving the fallback path is what's actually
exercised there) and once in the dedicated `test-real-backends` job that
installs `.[vectorstore,embeddings]` first (where these run for real,
against a real downloaded MiniLM model and a real FAISS index).
"""

import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ragent.vectorstore import FaissVectorStore

HAS_FAISS = importlib.util.find_spec("faiss") is not None
HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None


@unittest.skipUnless(HAS_FAISS, "faiss-cpu not installed (pip install ragent[vectorstore])")
class TestFaissVectorStore(unittest.TestCase):
    def test_add_and_search_matches_numpy_reference_ranking(self):
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(50, 16)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        ids = [f"vec{i}" for i in range(50)]

        store = FaissVectorStore(dim=16)
        store.add(ids, vectors)
        self.assertEqual(len(store), 50)

        # The vector closest to itself should always be itself.
        query = vectors[7]
        results = store.search(query, k=1)
        self.assertEqual(results[0].id, "vec7")
        self.assertAlmostEqual(results[0].score, 1.0, places=4)


@unittest.skipUnless(
    HAS_SENTENCE_TRANSFORMERS,
    "sentence-transformers not installed (pip install ragent[embeddings])",
)
class TestSentenceTransformerEmbedder(unittest.TestCase):
    def test_embeddings_are_normalized_and_capture_semantic_similarity(self):
        from ragent.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        vectors = embedder.embed(
            [
                "the cat sat on the mat",
                "a feline rested on the rug",
                "quarterly revenue grew by double digits",
            ]
        )
        self.assertEqual(vectors.shape[0], 3)
        for v in vectors:
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=3)

        sim_related = float(vectors[0] @ vectors[1])
        sim_unrelated = float(vectors[0] @ vectors[2])
        self.assertGreater(sim_related, sim_unrelated)


if __name__ == "__main__":
    unittest.main()
