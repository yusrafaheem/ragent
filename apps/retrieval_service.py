"""
Retrieval microservice: owns document ingestion, the embedding model, and
the vector index. Exposes /ingest and /search over HTTP so it can be scaled,
deployed, and versioned independently of the agent orchestrator service
(see apps/agent_service.py), which talks to this one exclusively through
its HTTP API -- never by importing its internals.

Run directly:
    uvicorn apps.retrieval_service:app --reload --port 8001

Or via Docker (see infra/Dockerfile.retrieval and docker-compose.yml).
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from ragent.retriever import Retriever

app = FastAPI(title="ragent-retrieval-service", version="0.1.0")

_retriever = Retriever(
    embedder_backend=os.environ.get("EMBEDDER_BACKEND", "auto"),
    vectorstore_backend=os.environ.get("VECTORSTORE_BACKEND", "auto"),
)


class IngestRequest(BaseModel):
    doc_id: str
    text: str
    chunk_size: int = 120
    overlap: int = 30


class IngestResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    total_chunks_in_store: int


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SearchResultItem(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks_indexed": len(_retriever)}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    before = len(_retriever)
    _retriever.ingest_document(
        req.doc_id, req.text, chunk_size=req.chunk_size, overlap=req.overlap
    )
    after = len(_retriever)
    return IngestResponse(
        doc_id=req.doc_id, chunks_indexed=after - before, total_chunks_in_store=after
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    results = _retriever.query(req.query, k=req.k)
    return SearchResponse(
        results=[
            SearchResultItem(
                doc_id=r.chunk.doc_id,
                chunk_id=r.chunk.chunk_id,
                text=r.chunk.text,
                score=r.score,
            )
            for r in results
        ]
    )
