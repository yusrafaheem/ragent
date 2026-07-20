"""
ragent.retriever
=================

Wires chunking + an Embedder + a VectorStore into a single ingest/query
pipeline, and keeps the chunk text/metadata needed to cite results (a raw
VectorStore only knows about ids and vectors, not the text behind them).
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk, chunk_text
from .embeddings import Embedder, get_embedder
from .vectorstore import VectorStore, get_vectorstore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        vectorstore: VectorStore | None = None,
        embedder_backend: str = "auto",
        vectorstore_backend: str = "auto",
    ):
        self.embedder = embedder or get_embedder(backend=embedder_backend)
        self.vectorstore = vectorstore or get_vectorstore(
            dim=self.embedder.dim, backend=vectorstore_backend
        )
        self._chunks_by_id: dict[str, Chunk] = {}

    def ingest_document(self, doc_id: str, text: str, chunk_size: int = 120, overlap: int = 30):
        chunks = chunk_text(doc_id, text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return
        vectors = self.embedder.embed([c.text for c in chunks])
        self.vectorstore.add([c.chunk_id for c in chunks], vectors)
        for c in chunks:
            self._chunks_by_id[c.chunk_id] = c

    def ingest_corpus(self, documents: dict[str, str], **kwargs) -> None:
        for doc_id, text in documents.items():
            self.ingest_document(doc_id, text, **kwargs)

    def query(self, question: str, k: int = 5) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed([question])[0]
        results = self.vectorstore.search(query_vector, k=k)
        return [
            RetrievedChunk(chunk=self._chunks_by_id[r.id], score=r.score)
            for r in results
            if r.id in self._chunks_by_id
        ]

    def __len__(self) -> int:
        return len(self.vectorstore)
