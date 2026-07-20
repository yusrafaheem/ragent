import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ragent.llm as llm_module
from ragent.llm import StubLLM, get_llm_client


class TestStubLLM(unittest.TestCase):
    def setUp(self):
        self.llm = StubLLM()

    def test_first_turn_emits_search_action(self):
        question = "Question: what is retrieval augmented generation?"
        messages = [{"role": "user", "content": question}]
        reply = self.llm.chat(messages)
        self.assertIn("Action: search_papers[", reply)
        self.assertIn("what is retrieval augmented generation?", reply)

    def test_after_observation_emits_final_answer(self):
        messages = [
            {"role": "user", "content": "Question: what is RAG?"},
            {"role": "assistant", "content": "Thought: ...\nAction: search_papers[RAG]"},
            {"role": "user", "content": "Observation: RAG grounds LLMs in retrieved documents."},
        ]
        reply = self.llm.chat(messages)
        self.assertIn("Final Answer:", reply)
        self.assertIn("RAG grounds LLMs in retrieved documents.", reply)


class TestGetLLMClient(unittest.TestCase):
    def test_explicit_stub_backend(self):
        self.assertIsInstance(get_llm_client(backend="stub"), StubLLM)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            get_llm_client(backend="not-a-real-backend")

    def test_auto_falls_back_to_stub_without_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            client = get_llm_client(backend="auto")
        self.assertIsInstance(client, StubLLM)

    def test_auto_uses_openai_backend_when_api_key_present(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake"}):
            with mock.patch.object(llm_module, "OpenAIChatLLM") as mock_cls:
                get_llm_client(backend="auto")
                mock_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
