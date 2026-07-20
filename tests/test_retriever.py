import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragent.retriever import Retriever


class TestRetriever(unittest.TestCase):
    def setUp(self):
        # Explicit "hashing" / "numpy" backends so this test is not
        # sensitive to whether the real ML extras happen to be installed.
        self.retriever = Retriever(embedder_backend="hashing", vectorstore_backend="numpy")

    def test_ingest_document_indexes_chunks(self):
        self.retriever.ingest_document(
            "paper1", "gradient descent optimizes neural network weights iteratively"
        )
        self.assertGreater(len(self.retriever), 0)

    def test_ingest_corpus_indexes_multiple_documents(self):
        self.retriever.ingest_corpus(
            {
                "paper1": "vector databases enable fast approximate nearest neighbor search",
                "paper2": "retrieval augmented generation grounds language models in documents",
            }
        )
        self.assertGreaterEqual(len(self.retriever), 2)

    def test_query_returns_most_relevant_document(self):
        self.retriever.ingest_corpus(
            {
                "vecdb_paper": (
                    "FAISS and other vector databases build indexes such as IVF and HNSW "
                    "to make nearest neighbor search over embeddings sub-linear."
                ),
                "unrelated_paper": (
                    "quarterly earnings reports showed strong revenue growth across the "
                    "retail sector this fiscal year."
                ),
            }
        )

        results = self.retriever.query("how do vector databases index embeddings for search", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.doc_id, "vecdb_paper")

    def test_empty_corpus_query_returns_no_results(self):
        results = self.retriever.query("anything", k=5)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
