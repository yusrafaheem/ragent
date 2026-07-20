"""
ragent.tools
============

Tools the agent can invoke mid-reasoning. A ``Tool`` is just a name, a
description (shown to the LLM so it knows the tool exists and what to pass
it), and a callable that takes a single string argument and returns a
string observation.

Tool calls are parsed out of free-form LLM text (see agent.py's
``ACTION_RE``), so every tool function must be safe to call with
attacker/model-controlled input -- in particular, ``calculator`` below
evaluates arithmetic via Python's ``ast`` module rather than ``eval()``, so
it cannot execute arbitrary code no matter what string the model produces.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .retriever import Retriever

if TYPE_CHECKING:
    import httpx

_ALLOWED_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    fn: Callable[[str], str]

    def __call__(self, arg: str) -> str:
        return self.fn(arg)


def _safe_eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a numeric arithmetic expression safely (no ``eval``)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_node(tree.body)
        return str(result)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any error becomes an observation
        return f"error: could not evaluate '{expression}': {exc}"


CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Evaluate a numeric arithmetic expression, e.g. '2 * (3 + 4)'.",
    fn=calculator,
)


def make_search_papers_tool(retriever: Retriever, k: int = 3) -> Tool:
    """Build a search_papers tool bound to a specific Retriever instance."""

    def _search(query: str) -> str:
        results = retriever.query(query, k=k)
        if not results:
            return "no results found"
        lines = []
        for r in results:
            snippet = r.chunk.text[:280]
            lines.append(f"[{r.chunk.doc_id} score={r.score:.3f}] {snippet}")
        return " | ".join(lines)

    return Tool(
        name="search_papers",
        description=(
            "Search the ingested paper corpus for passages relevant to a query. "
            "Argument: a natural-language search query."
        ),
        fn=_search,
    )


def make_remote_search_papers_tool(
    base_url: str, k: int = 3, client: "httpx.Client | None" = None
) -> Tool:
    """Build a search_papers tool that calls a retrieval microservice over
    HTTP, instead of an in-process Retriever.

    This is what the agent service uses in production: the retrieval index
    lives in a separate process (and separate container/pod) from the agent
    orchestrator, so this is a genuine network call and failure boundary,
    not a function call dressed up as one.

    httpx is imported lazily here, and only if the caller didn't already
    supply a client (tests inject a mock/duck-typed client instead), so that
    importing ragent.tools -- and therefore ragent, since __init__.py
    exposes Agent -- doesn't hard-require httpx to be installed just to use
    the in-process search_papers tool or the calculator tool. httpx is a
    core dependency for actually running the agent service, but the core
    package and its test suite stay exercisable with nothing beyond NumPy.
    Failures are caught as a plain Exception (rather than httpx.HTTPError
    specifically) for the same reason: it keeps this function correct
    against any object satisfying the "has .post()" duck type, not just a
    real httpx.Client.
    """
    if client is None:
        import httpx

        client = httpx.Client(base_url=base_url, timeout=10.0)
    http_client = client

    def _search(query: str) -> str:
        try:
            response = http_client.post("/search", json={"query": query, "k": k})
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - any transport/HTTP failure becomes an observation
            return f"error: retrieval service request failed: {exc}"

        results = response.json().get("results", [])
        if not results:
            return "no results found"
        lines = [
            f"[{r['doc_id']} score={r['score']:.3f}] {r['text'][:280]}" for r in results
        ]
        return " | ".join(lines)

    return Tool(
        name="search_papers",
        description=(
            "Search the ingested paper corpus (via the retrieval service) for "
            "passages relevant to a query. Argument: a natural-language search query."
        ),
        fn=_search,
    )
