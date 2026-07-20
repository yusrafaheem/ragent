import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragent.retriever import Retriever
from ragent.tools import (
    CALCULATOR_TOOL,
    calculator,
    make_remote_search_papers_tool,
    make_search_papers_tool,
)


class TestCalculator(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(calculator("2 + 3"), "5")
        self.assertEqual(calculator("2 * (3 + 4)"), "14")
        self.assertEqual(calculator("10 / 4"), "2.5")
        self.assertEqual(calculator("2 ** 10"), "1024")

    def test_negative_numbers(self):
        self.assertEqual(calculator("-5 + 3"), "-2")

    def test_rejects_arbitrary_code_execution(self):
        result = calculator("__import__('os').system('echo pwned')")
        self.assertTrue(result.startswith("error:"))

    def test_rejects_non_numeric_names(self):
        result = calculator("open('/etc/passwd').read()")
        self.assertTrue(result.startswith("error:"))

    def test_calculator_tool_wraps_function(self):
        self.assertEqual(CALCULATOR_TOOL.name, "calculator")
        self.assertEqual(CALCULATOR_TOOL("1 + 1"), "2")


class TestSearchPapersTool(unittest.TestCase):
    def test_returns_no_results_found_on_empty_index(self):
        retriever = Retriever(embedder_backend="hashing", vectorstore_backend="numpy")
        tool = make_search_papers_tool(retriever)
        self.assertEqual(tool("anything"), "no results found")

    def test_returns_formatted_results_with_doc_id_and_score(self):
        retriever = Retriever(embedder_backend="hashing", vectorstore_backend="numpy")
        retriever.ingest_document("paper1", "transformers use self-attention over input tokens")
        tool = make_search_papers_tool(retriever, k=1)

        result = tool("what mechanism do transformers use")
        self.assertIn("paper1", result)
        self.assertIn("score=", result)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class TestRemoteSearchPapersTool(unittest.TestCase):
    def test_formats_remote_results(self):
        fake_client = mock.Mock()
        fake_client.post.return_value = _FakeResponse(
            {
                "results": [
                    {
                        "doc_id": "paper1",
                        "chunk_id": "paper1::chunk0",
                        "text": "some passage",
                        "score": 0.87,
                    }
                ]
            }
        )
        tool = make_remote_search_papers_tool("http://retrieval:8001", client=fake_client)

        result = tool("a query")
        self.assertIn("paper1", result)
        self.assertIn("some passage", result)
        fake_client.post.assert_called_once_with(
            "/search", json={"query": "a query", "k": 3}
        )

    def test_no_results_from_remote_service(self):
        fake_client = mock.Mock()
        fake_client.post.return_value = _FakeResponse({"results": []})
        tool = make_remote_search_papers_tool("http://retrieval:8001", client=fake_client)
        self.assertEqual(tool("a query"), "no results found")

    def test_transport_error_surfaces_as_observation_not_exception(self):
        # A connection failure (real httpx raises httpx.ConnectError here;
        # any Exception works for this test since make_remote_search_papers_tool
        # deliberately catches broadly -- see its docstring) must become a
        # returned observation string, never propagate and crash the agent loop.
        fake_client = mock.Mock()
        fake_client.post.side_effect = ConnectionError("connection refused")
        tool = make_remote_search_papers_tool("http://retrieval:8001", client=fake_client)

        result = tool("a query")
        self.assertTrue(result.startswith("error:"))


if __name__ == "__main__":
    unittest.main()
