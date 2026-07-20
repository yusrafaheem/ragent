"""
ragent.embeddings
==================

Pluggable text-embedding backends behind a single ``Embedder`` interface.

Design note: this project targets environments that may not have network
access to install or call a real embedding model (sentence-transformers
requires a multi-hundred-MB model download; OpenAI's embeddings API requires
a paid key and network egress). Rather than making the whole system
undemoable without those, every backend implements the same
``embed(texts) -> np.ndarray`` contract, and ``get_embedder`` tries the real
backend first and falls back to a zero-dependency hashing embedder with a
warning. This mirrors how the retrieval and LLM layers are structured, and
lets CI exercise the full RAG pipeline offline while still supporting a real
transformer backend wherever one is installed (see the ``embeddings`` extra
in pyproject.toml, and the ``embeddings-real`` CI job that installs it).
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    """Common interface for turning text into fixed-size vectors."""

    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), self.dim) float32 array of L2-normalized
        embeddings.
        """
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Zero-dependency bag-of-words embedding via the hashing trick.

    Each token is hashed into one of ``dim`` buckets (with a sign derived
    from a second hash, to make the estimator roughly unbiased -- the same
    trick scikit-learn's ``HashingVectorizer`` uses), the resulting sparse
    vector is L2-normalized, and cosine similarity search over these vectors
    behaves like a bag-of-words / TF-style retriever. It has no notion of
    word meaning (no synonyms, no context), which is precisely the gap a
    real transformer embedding model closes -- but it needs no downloads, no
    GPU, and no network, which makes it a useful reference implementation
    and CI fallback.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _hash_token(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            for token in tokens:
                index, sign = self._hash_token(token)
                out[row, index] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder(Embedder):
    """Wraps a real sentence-transformers model, if installed.

    Raises ImportError with an actionable message otherwise, so callers get
    a clear signal instead of a confusing downstream failure.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only when installed
            raise ImportError(
                "SentenceTransformerEmbedder requires the 'sentence-transformers' "
                "package. Install with: pip install ragent[embeddings]"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs the real model
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(backend: str = "auto", dim: int = 512) -> Embedder:
    """Resolve an embedding backend by name.

    backend="auto" prefers sentence-transformers if it's importable, and
    falls back to the hashing embedder (logging why) otherwise.
    """
    if backend == "hashing":
        return HashingEmbedder(dim=dim)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    if backend != "auto":
        raise ValueError(f"unknown embedder backend: {backend}")

    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        logger.info(
            "sentence-transformers not installed; falling back to HashingEmbedder. "
            "Install 'ragent[embeddings]' for real dense embeddings."
        )
        return HashingEmbedder(dim=dim)
