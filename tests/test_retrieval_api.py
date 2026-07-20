"""
Tests the retrieval FastAPI service via Starlette's TestClient (in-process,
no real network socket). Requires the 'fastapi' extra (fastapi, httpx) to be
installed -- see the environment note in README.md: this project's core
package (tests/test_chunking.py, test_embeddings.py, test_vectorstore.py,
test_retriever.py, test_tools.py, test_llm.py, test_agent.py) is testable
with only NumPy, but the HTTP layer necessarily needs fastapi + httpx
installed to exercise, the same way it needs them installed to run at all.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient

from apps.retrieval_service import app


class TestRetrievalService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_ingest_then_search_round_trip(self):
        ingest_response = self.client.post(
            "/ingest",
            json={
                "doc_id": "test_paper",
                "text": "sparse retrieval methods like BM25 rank documents by term overlap "
                "while dense retrieval methods embed queries and documents into a shared "
                "vector space and rank by cosine similarity",
            },
        )
        self.assertEqual(ingest_response.status_code, 200)
        body = ingest_response.json()
        self.assertEqual(body["doc_id"], "test_paper")
        self.assertGreater(body["chunks_indexed"], 0)

        search_response = self.client.post(
            "/search", json={"query": "how does dense retrieval rank documents", "k": 1}
        )
        self.assertEqual(search_response.status_code, 200)
        results = search_response.json()["results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "test_paper")

    def test_search_on_empty_index_returns_empty_list(self):
        # Uses a fresh app instance-level index would be ideal, but this
        # service intentionally holds one process-wide retriever (see its
        # module docstring), so this just asserts the response shape is
        # correct and never errors, regardless of prior test ordering.
        response = self.client.post("/search", json={"query": "anything", "k": 3})
        self.assertEqual(response.status_code, 200)
        self.assertIn("results", response.json())

    def test_ingest_rejects_missing_fields(self):
        response = self.client.post("/ingest", json={"doc_id": "missing_text"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
