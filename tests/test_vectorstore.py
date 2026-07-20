import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import ragent.vectorstore as vectorstore_module
from ragent.vectorstore import NumpyVectorStore, get_vectorstore


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


class TestNumpyVectorStore(unittest.TestCase):
    def test_empty_store_returns_no_results(self):
        store = NumpyVectorStore(dim=4)
        self.assertEqual(len(store), 0)
        self.assertEqual(store.search(np.array([1.0, 0.0, 0.0, 0.0])), [])

    def test_add_and_search_returns_exact_match_first(self):
        store = NumpyVectorStore(dim=3)
        vectors = np.array(
            [
                _unit(np.array([1.0, 0.0, 0.0])),
                _unit(np.array([0.0, 1.0, 0.0])),
                _unit(np.array([0.0, 0.0, 1.0])),
            ]
        )
        store.add(["x", "y", "z"], vectors)
        self.assertEqual(len(store), 3)

        results = store.search(_unit(np.array([1.0, 0.01, 0.0])), k=1)
        self.assertEqual(results[0].id, "x")

    def test_search_results_are_sorted_descending_by_score(self):
        store = NumpyVectorStore(dim=2)
        vectors = np.array(
            [
                _unit(np.array([1.0, 0.0])),
                _unit(np.array([0.9, 0.1])),
                _unit(np.array([-1.0, 0.0])),
            ]
        )
        store.add(["best", "second", "worst"], vectors)

        results = store.search(np.array([1.0, 0.0]), k=3)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(results[0].id, "best")
        self.assertEqual(results[-1].id, "worst")

    def test_k_larger_than_store_size_does_not_crash(self):
        store = NumpyVectorStore(dim=2)
        store.add(["a"], np.array([_unit(np.array([1.0, 1.0]))]))
        results = store.search(np.array([1.0, 0.0]), k=10)
        self.assertEqual(len(results), 1)

    def test_mismatched_dim_raises(self):
        store = NumpyVectorStore(dim=4)
        with self.assertRaises(ValueError):
            store.add(["a"], np.zeros((1, 3)))

    def test_mismatched_ids_and_vectors_length_raises(self):
        store = NumpyVectorStore(dim=3)
        with self.assertRaises(ValueError):
            store.add(["a", "b"], np.zeros((1, 3)))

    def test_multiple_add_calls_accumulate(self):
        store = NumpyVectorStore(dim=2)
        store.add(["a"], np.array([_unit(np.array([1.0, 0.0]))]))
        store.add(["b"], np.array([_unit(np.array([0.0, 1.0]))]))
        self.assertEqual(len(store), 2)


class TestGetVectorstore(unittest.TestCase):
    def test_explicit_numpy_backend(self):
        store = get_vectorstore(dim=8, backend="numpy")
        self.assertIsInstance(store, NumpyVectorStore)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            get_vectorstore(dim=8, backend="not-a-real-backend")

    def test_auto_falls_back_to_numpy_when_faiss_unavailable(self):
        with mock.patch.object(
            vectorstore_module, "FaissVectorStore", side_effect=ImportError("forced for test")
        ):
            store = get_vectorstore(dim=8, backend="auto")
        self.assertIsInstance(store, NumpyVectorStore)


if __name__ == "__main__":
    unittest.main()
