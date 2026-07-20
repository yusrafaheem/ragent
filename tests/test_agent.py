import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragent.agent import Agent
from ragent.llm import LLMClient, StubLLM
from ragent.retriever import Retriever
from ragent.tools import CALCULATOR_TOOL, make_search_papers_tool


class ScriptedLLM(LLMClient):
    """Replays a fixed sequence of responses, one per call -- for testing
    exact multi-turn agent behavior without depending on StubLLM's specific
    parsing rules.
    """

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self._responses[len(self.calls) - 1]


class TestAgentWithStubLLM(unittest.TestCase):
    def test_full_loop_search_then_answer(self):
        retriever = Retriever(embedder_backend="hashing", vectorstore_backend="numpy")
        retriever.ingest_document(
            "rag_paper",
            "retrieval augmented generation grounds a language model's answers in "
            "passages retrieved from an external document index at inference time",
        )
        tools = [make_search_papers_tool(retriever, k=1)]
        agent = Agent(llm=StubLLM(), tools=tools, max_steps=4)

        result = agent.run("Question: what does retrieval augmented generation do?")

        self.assertFalse(result.hit_max_steps)
        self.assertIn("rag_paper", result.answer)
        # Exactly one search step, then a final-answer step.
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0].action, "search_papers")
        self.assertIsNotNone(result.steps[0].observation)
        self.assertIsNone(result.steps[1].action)


class TestAgentWithScriptedLLM(unittest.TestCase):
    def test_calculator_tool_is_invoked_and_result_used(self):
        responses = [
            "Thought: I need to compute this.\nAction: calculator[6 * 7]",
            "Thought: got it.\nFinal Answer: the result is 42",
        ]
        agent = Agent(llm=ScriptedLLM(responses), tools=[CALCULATOR_TOOL], max_steps=4)

        result = agent.run("Question: what is 6 times 7?")

        self.assertEqual(result.answer, "the result is 42")
        self.assertEqual(result.steps[0].action, "calculator")
        self.assertEqual(result.steps[0].observation, "42")

    def test_unknown_tool_name_produces_error_observation_not_crash(self):
        responses = [
            "Thought: hmm.\nAction: nonexistent_tool[foo]",
            "Thought: giving up gracefully.\nFinal Answer: could not complete the task",
        ]
        agent = Agent(llm=ScriptedLLM(responses), tools=[CALCULATOR_TOOL], max_steps=4)

        result = agent.run("Question: anything")

        self.assertIn("unknown tool", result.steps[0].observation)
        self.assertEqual(result.answer, "could not complete the task")

    def test_hits_max_steps_if_llm_never_produces_final_answer(self):
        # Every response is another action, so the agent should stop after
        # max_steps rather than looping forever.
        responses = ["Thought: still working.\nAction: calculator[1 + 1]" for _ in range(10)]
        agent = Agent(llm=ScriptedLLM(responses), tools=[CALCULATOR_TOOL], max_steps=3)

        result = agent.run("Question: anything")

        self.assertTrue(result.hit_max_steps)
        self.assertEqual(len(result.steps), 3)

    def test_unparseable_response_is_returned_verbatim_as_answer(self):
        responses = ["I'm not following the required format at all."]
        agent = Agent(llm=ScriptedLLM(responses), tools=[CALCULATOR_TOOL], max_steps=4)

        result = agent.run("Question: anything")

        self.assertEqual(result.answer, "I'm not following the required format at all.")
        self.assertFalse(result.hit_max_steps)

    def test_tool_descriptions_are_included_in_prompt(self):
        responses = ["Final Answer: n/a"]
        llm = ScriptedLLM(responses)
        agent = Agent(llm=llm, tools=[CALCULATOR_TOOL], max_steps=1)

        agent.run("Question: anything")

        first_prompt = llm.calls[0][0]["content"]
        self.assertIn("calculator", first_prompt)
        self.assertIn("Question: anything", first_prompt)


if __name__ == "__main__":
    unittest.main()
