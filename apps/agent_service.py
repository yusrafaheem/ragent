"""
Agent orchestrator microservice: owns the LLM client and the ReAct loop.
Holds no document index of its own -- to answer a question it calls the
retrieval service's HTTP API as a tool (see
ragent.tools.make_remote_search_papers_tool), the same way it would call
any other external service. This split mirrors a real production layout:
the retrieval index is stateful, potentially large, and expensive to scale
(it owns the embedding model and vector index); the agent orchestrator is
stateless and horizontally scalable independent of it.

Run directly (with the retrieval service already running):
    RETRIEVAL_SERVICE_URL=http://localhost:8001 \\
        uvicorn apps.agent_service:app --reload --port 8000

Or via Docker (see infra/Dockerfile.agent and docker-compose.yml).
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from ragent.agent import Agent, AgentStep
from ragent.llm import get_llm_client
from ragent.tools import CALCULATOR_TOOL, make_remote_search_papers_tool

app = FastAPI(title="ragent-agent-service", version="0.1.0")

_retrieval_url = os.environ.get("RETRIEVAL_SERVICE_URL", "http://localhost:8001")
_llm = get_llm_client(backend=os.environ.get("LLM_BACKEND", "auto"))
_tools = [make_remote_search_papers_tool(_retrieval_url), CALCULATOR_TOOL]
_agent = Agent(llm=_llm, tools=_tools, max_steps=int(os.environ.get("AGENT_MAX_STEPS", "6")))


class AskRequest(BaseModel):
    question: str


class AgentStepView(BaseModel):
    thought: str
    action: str | None
    action_input: str | None
    observation: str | None


class AskResponse(BaseModel):
    answer: str
    hit_max_steps: bool
    steps: list[AgentStepView]


def _view(step: AgentStep) -> AgentStepView:
    return AgentStepView(
        thought=step.thought,
        action=step.action,
        action_input=step.action_input,
        observation=step.observation,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "retrieval_service_url": _retrieval_url}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = _agent.run(req.question)
    return AskResponse(
        answer=result.answer,
        hit_max_steps=result.hit_max_steps,
        steps=[_view(s) for s in result.steps],
    )
