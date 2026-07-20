"""
Tests the agent orchestrator FastAPI service via Starlette's TestClient.
See test_retrieval_api.py's module docstring for the dependency note.

The module-level ``_agent`` in apps.agent_service is built against the real
LLM/tool backends at import time (so the service works out of the box when
actually deployed), so these tests swap it out for a deterministic scripted
agent before each request -- exactly the same "swap the LLM backend behind
a fixed interface" pattern used by ragent.agent's own test suite, just
applied at the HTTP layer instead of calling Agent.run() directly.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient

import apps.agent_service as agent_service_module
from apps.agent_service import app
from ragent.agent import Agent
from ragent.llm import LLMClient
from ragent.tools import CALCULATOR_TOOL


class _ScriptedLLM(LLMClient):
    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def chat(self, messages):
        reply = self._responses[self._i]
        self._i = min(self._i + 1, len(self._responses) - 1)
        return reply


class TestAgentService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_ask_returns_final_answer_and_step_trace(self):
        scripted_agent = Agent(
            llm=_ScriptedLLM(
                [
                    "Thought: compute it.\nAction: calculator[21 * 2]",
                    "Thought: done.\nFinal Answer: the answer is 42",
                ]
            ),
            tools=[CALCULATOR_TOOL],
            max_steps=4,
        )

        with mock.patch.object(agent_service_module, "_agent", scripted_agent):
            response = self.client.post("/ask", json={"question": "what is 21 times 2?"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "the answer is 42")
        self.assertFalse(body["hit_max_steps"])
        self.assertEqual(len(body["steps"]), 2)
        self.assertEqual(body["steps"][0]["action"], "calculator")
        self.assertEqual(body["steps"][0]["observation"], "42")

    def test_ask_rejects_missing_question_field(self):
        response = self.client.post("/ask", json={})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
