"""
ragent.llm
==========

Pluggable chat-completion backend behind a single ``LLMClient`` interface,
following the same real-backend-with-offline-fallback pattern as
embeddings.py and vectorstore.py.

``StubLLM`` is not a mock used only in tests -- it's a real, if simple,
rule-based ReAct policy: it recognizes the agent's prompt format (see
agent.py) well enough to decide when to call the search_papers tool versus
emit a final answer, deterministically. That matters for two reasons: the
CI pipeline can run the *entire* agent loop end-to-end (prompt formatting,
tool dispatch, observation handling, answer synthesis) without an API key or
network access, and the agent/tool-dispatch logic can be unit-tested
deterministically instead of asserting on non-reproducible LLM output.
``OpenAIChatLLM`` is the real backend, used whenever OPENAI_API_KEY is set.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Common interface: send chat messages, get back assistant text."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


class StubLLM(LLMClient):
    """Deterministic offline ReAct policy.

    Looks at the most recent messages to decide the next step:
      - If there's no "Observation:" yet in the transcript, emit an Action
        that calls the search_papers tool with the user's question as the
        query.
      - If there is an Observation, synthesize a Final Answer that quotes
        the retrieved chunk text directly (a real LLM would paraphrase and
        reason over it; this stub proves the tool-calling plumbing works).
    """

    def chat(self, messages: list[dict[str, str]]) -> str:
        transcript = "\n".join(m["content"] for m in messages)

        if "Observation:" not in transcript:
            question_match = re.search(r"Question:\s*(.+)", transcript)
            question = question_match.group(1).strip() if question_match else ""
            return (
                "Thought: I should search the paper corpus for relevant passages.\n"
                f'Action: search_papers[{question}]'
            )

        observations = re.findall(r"Observation:\s*(.+)", transcript)
        latest = observations[-1] if observations else "no results found"
        return (
            "Thought: I have enough evidence to answer.\n"
            f"Final Answer: Based on the retrieved passages: {latest}"
        )


class OpenAIChatLLM(LLMClient):
    """Wraps the OpenAI-compatible chat completions API.

    Also works against any OpenAI-API-compatible endpoint (e.g. a
    self-hosted vLLM server, or Gemini's OpenAI-compatibility layer) by
    setting OPENAI_BASE_URL alongside OPENAI_API_KEY.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only when installed
            raise ImportError(
                "OpenAIChatLLM requires the 'openai' package. "
                "Install with: pip install ragent[llm]"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Either set it, or use StubLLM / "
                "get_llm_client('stub') for offline use."
            )

        base_url = os.environ.get("OPENAI_BASE_URL")
        self._client = OpenAI(api_key=key, base_url=base_url)
        self.model = model

    def chat(self, messages: list[dict[str, str]]) -> str:  # pragma: no cover - needs a real key
        response = self._client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content or ""


def get_llm_client(backend: str = "auto", model: str = "gpt-4o-mini") -> LLMClient:
    """Resolve an LLM backend by name.

    backend="auto" uses OpenAIChatLLM if OPENAI_API_KEY is set in the
    environment, and falls back to StubLLM otherwise.
    """
    if backend == "stub":
        return StubLLM()
    if backend == "openai":
        return OpenAIChatLLM(model=model)
    if backend != "auto":
        raise ValueError(f"unknown LLM backend: {backend}")

    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIChatLLM(model=model)
    return StubLLM()
