"""
ragent.agent
============

A small ReAct-style (Reason + Act) agent loop: the LLM alternates between
emitting a ``Thought`` + ``Action: tool_name[argument]`` pair and reading
back an ``Observation`` from the tool it called, until it emits a
``Final Answer``. This is the same pattern used by LangChain's ReAct agent
and the original ReAct paper (Yao et al., 2022) -- implemented from scratch
here (no agent framework dependency) so the control flow is fully
inspectable and testable.

The loop is intentionally backend-agnostic: it drives any ``LLMClient``
(StubLLM or a real API-backed one) through the exact same code path, which
is what lets the test suite assert on agent *behavior* (does it call the
right tool, does it stop within max_steps, does it surface the observation
in its answer) without depending on a specific model's non-deterministic
output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLMClient
from .tools import Tool

_ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)

_SYSTEM_PROMPT_TEMPLATE = """You are a research assistant that answers questions using the tools \
below. At each step, respond with either:

Thought: <your reasoning>
Action: <tool_name>[<argument>]

or, once you have enough information:

Thought: <your reasoning>
Final Answer: <your answer, citing which document(s) it came from>

Available tools:
{tool_descriptions}

Question: {question}"""


@dataclass
class AgentStep:
    thought: str
    action: str | None
    action_input: str | None
    observation: str | None


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    hit_max_steps: bool = False


class Agent:
    def __init__(self, llm: LLMClient, tools: list[Tool], max_steps: int = 6):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps

    def _tool_descriptions(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

    def run(self, question: str) -> AgentResult:
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=self._tool_descriptions(), question=question
        )
        messages: list[dict[str, str]] = [{"role": "user", "content": system_prompt}]
        steps: list[AgentStep] = []

        for _ in range(self.max_steps):
            reply = self.llm.chat(messages)
            messages.append({"role": "assistant", "content": reply})

            final_match = _FINAL_ANSWER_RE.search(reply)
            if final_match:
                steps.append(
                    AgentStep(thought=reply, action=None, action_input=None, observation=None)
                )
                return AgentResult(answer=final_match.group(1).strip(), steps=steps)

            action_match = _ACTION_RE.search(reply)
            if not action_match:
                # The model didn't follow the format; surface its raw text as
                # the answer rather than looping forever on unparseable output.
                steps.append(
                    AgentStep(thought=reply, action=None, action_input=None, observation=None)
                )
                return AgentResult(answer=reply.strip(), steps=steps)

            tool_name, tool_arg = action_match.group(1), action_match.group(2).strip()
            tool = self.tools.get(tool_name)
            if tool is None:
                observation = f"error: unknown tool '{tool_name}'"
            else:
                observation = tool(tool_arg)

            steps.append(
                AgentStep(
                    thought=reply, action=tool_name, action_input=tool_arg, observation=observation
                )
            )
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return AgentResult(
            answer="I was unable to reach a final answer within the step budget.",
            steps=steps,
            hit_max_steps=True,
        )
