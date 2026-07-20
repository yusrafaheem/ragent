"""
Ingest every .txt file in a directory into a running retrieval service.

Usage:
    python scripts/ingest_corpus.py --dir data/sample_papers --url http://localhost:8001

Requires the retrieval service to already be running (see README.md for how
to start it directly with uvicorn or via docker-compose).
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/sample_papers", help="directory of .txt files")
    parser.add_argument("--url", default="http://localhost:8001", help="retrieval service base URL")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"error: {args.dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    txt_files = sorted(f for f in os.listdir(args.dir) if f.endswith(".txt"))
    if not txt_files:
        print(f"no .txt files found in {args.dir}")
        return

    client = httpx.Client(base_url=args.url, timeout=30.0)
    try:
        client.get("/health").raise_for_status()
    except httpx.HTTPError as exc:
        print(f"error: could not reach retrieval service at {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)

    for filename in txt_files:
        doc_id = os.path.splitext(filename)[0]
        with open(os.path.join(args.dir, filename), encoding="utf-8") as f:
            text = f.read()

        response = client.post("/ingest", json={"doc_id": doc_id, "text": text})
        response.raise_for_status()
        body = response.json()
        print(
            f"ingested {doc_id}: {body['chunks_indexed']} chunks "
            f"({body['total_chunks_in_store']} total in store)"
        )


if __name__ == "__main__":
    main()
