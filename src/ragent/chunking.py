"""
ragent.chunking
================

Splits raw document text into overlapping, retrieval-sized chunks.

Overlap matters for RAG quality: a naive non-overlapping split can cut a
sentence's supporting evidence into a different chunk than its claim, so a
top-k retrieval can miss the passage that actually answers the question. This
module uses a sliding window over whitespace-tokenized words with a
configurable overlap, which is simple, dependency-free, and good enough for
plain-text scientific abstracts -- a production system would swap this for a
sentence- or section-aware splitter without touching any other module, since
callers only depend on the ``Chunk`` dataclass and ``chunk_text`` signature.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit of text, plus enough metadata to trace it
    back to its source document for citation.
    """

    doc_id: str
    chunk_id: str
    text: str
    start_word: int
    end_word: int


def chunk_text(
    doc_id: str,
    text: str,
    chunk_size: int = 120,
    overlap: int = 30,
) -> list[Chunk]:
    """Split ``text`` into overlapping word-window chunks.

    Parameters
    ----------
    doc_id:
        Identifier of the source document, propagated onto every chunk so
        retrieval results can be cited back to it.
    text:
        Raw document text.
    chunk_size:
        Number of words per chunk.
    overlap:
        Number of words shared between consecutive chunks. Must be smaller
        than ``chunk_size`` or the window never advances.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    words = text.split()
    if not words:
        return []

    stride = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::chunk{index}",
                text=" ".join(chunk_words),
                start_word=start,
                end_word=end,
            )
        )
        index += 1
        if end == len(words):
            break
        start += stride

    return chunks
