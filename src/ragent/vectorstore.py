"""
ragent.vectorstore
===================

Pluggable nearest-neighbor vector search behind a single ``VectorStore``
interface, with the same "real backend with a dependency-free fallback"
pattern used throughout this project (see embeddings.py, llm.py).

``NumpyVectorStore`` is a brute-force reference implementation: every query
is an O(n) matrix-vector product against every stored vector. That is
completely fine at the corpus sizes a demo or a single research group's
paper library needs (thousands of chunks), and it needs nothing beyond
NumPy. It stops being fine somewhere in the 10^5-10^7 vector range, which is
exactly the regime FAISS's indexing structures (IVF, HNSW, product
quantization) exist to make sub-linear -- ``FaissVectorStore`` wraps a real
FAISS flat index when the optional dependency is installed, as a drop-in
replacement behind the same interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    id: str
    score: float


class VectorStore(ABC):
    """Common interface for adding vectors and running top-k similarity
    search over them.
    """

    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class NumpyVectorStore(VectorStore):
    """Brute-force cosine-similarity search using a single NumPy matrix.

    Assumes input vectors are already L2-normalized (every embedder in this
    project normalizes its output), so cosine similarity reduces to a plain
    dot product and top-k search is one matrix-vector multiply plus an
    argpartition.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[str] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors must have the same length")

        self._ids.extend(ids)
        self._matrix = (
            vectors if self._matrix.shape[0] == 0 else np.vstack([self._matrix, vectors])
        )

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        if len(self._ids) == 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        scores = self._matrix @ query
        k = min(k, len(self._ids))
        # argpartition gives the top-k unordered in O(n); we only sort those.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [SearchResult(id=self._ids[i], score=float(scores[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._ids)


class FaissVectorStore(VectorStore):
    """Wraps a real ``faiss.IndexFlatIP`` (exact inner-product search), if
    the optional ``faiss-cpu`` dependency is installed.

    Kept API-compatible with NumpyVectorStore so the two are interchangeable
    everywhere else in the codebase -- see get_vectorstore().
    """

    def __init__(self, dim: int):
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - exercised only when installed
            raise ImportError(
                "FaissVectorStore requires the 'faiss-cpu' package. "
                "Install with: pip install ragent[vectorstore]"
            ) from exc

        self.dim = dim
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []

    def add(self, ids: list[str], vectors: np.ndarray) -> None:  # pragma: no cover
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        self._index.add(vectors)
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:  # pragma: no cover
        if len(self._ids) == 0:
            return []
        query = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        k = min(k, len(self._ids))
        scores, indices = self._index.search(query, k)
        return [
            SearchResult(id=self._ids[i], score=float(s))
            for s, i in zip(scores[0], indices[0])
            if i != -1
        ]

    def __len__(self) -> int:  # pragma: no cover
        return len(self._ids)


def get_vectorstore(dim: int, backend: str = "auto") -> VectorStore:
    """Resolve a vector store backend by name.

    backend="auto" prefers FAISS if it's importable, and falls back to the
    NumPy reference implementation (logging why) otherwise.
    """
    if backend == "numpy":
        return NumpyVectorStore(dim=dim)
    if backend == "faiss":
        return FaissVectorStore(dim=dim)
    if backend != "auto":
        raise ValueError(f"unknown vector store backend: {backend}")

    try:
        return FaissVectorStore(dim=dim)
    except ImportError:
        logger.info(
            "faiss-cpu not installed; falling back to NumpyVectorStore. "
            "Install 'ragent[vectorstore]' for indexed sub-linear search."
        )
        return NumpyVectorStore(dim=dim)
