"""
Ask the running agent service a question and pretty-print its full ReAct
trace (thought/action/observation steps) plus the final answer.

Usage:
    python scripts/demo.py "What does retrieval augmented generation do?"
    python scripts/demo.py --url http://localhost:8000 "What is HNSW?"

Requires both the retrieval service and the agent service to be running,
with the retrieval corpus already ingested (see scripts/ingest_corpus.py).
"""

from __future__ import annotations

import argparse
import sys

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--url", default="http://localhost:8000", help="agent service base URL")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.url, timeout=60.0)
    try:
        response = client.post("/ask", json={"question": args.question})
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"error: request to agent service at {args.url} failed: {exc}", file=sys.stderr)
        sys.exit(1)

    body = response.json()

    print(f"Question: {args.question}\n")
    for i, step in enumerate(body["steps"], start=1):
        print(f"--- step {i} ---")
        print(step["thought"].strip())
        if step["observation"] is not None:
            print(f"Observation: {step['observation']}")
        print()

    if body["hit_max_steps"]:
        print("(agent hit its step budget without a confident final answer)\n")

    print(f"Answer: {body['answer']}")


if __name__ == "__main__":
    main()
