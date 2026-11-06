"""Retriever for the KoBioMed-Llama RAG pipeline.

The retriever is the part that fetches: given a natural-language
question, it embeds the question with ``embeddings.Embedder``, asks the
``vectorstore.VectorStore`` for the nearest chunks, and returns them as
citable results — each hit knows which paper and page (or PMID) it came
from, so the answer generator can cite its sources instead of asking
the reader to trust it.

This module deliberately knows nothing about language models: it stops
at "here are the k most relevant passages, with provenance". Prompt
assembly lives in ``prompts.py``; generation lives downstream.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from embeddings import Embedder
from vectorstore import SearchHit, VectorStore

#: Default number of chunks returned per query.
DEFAULT_K = 5


@dataclass
class RetrievedChunk:
    """A chunk fetched for a query, with score and a ready citation."""

    chunk: dict
    score: float

    @property
    def chunk_id(self) -> str:
        return str(self.chunk.get("chunk_id", ""))

    @property
    def text(self) -> str:
        return str(self.chunk.get("text", ""))

    @property
    def citation(self) -> str:
        """Human-readable source label, e.g. ``PMID:38291023`` or
        ``park2026.pdf, p. 4``. Falls back to the raw source string."""
        metadata = self.chunk.get("metadata", {}) or {}
        pmid = metadata.get("pmid") or ""
        if not pmid and str(self.chunk.get("source", "")).startswith("PMID:"):
            pmid = str(self.chunk["source"]).split(":", 1)[1]
        if pmid:
            doi = metadata.get("doi", "")
            return f"PMID:{pmid}" + (f" (doi:{doi})" if doi else "")
        source = str(self.chunk.get("source", "unknown"))
        page = self.chunk.get("page")
        return f"{source}, p. {page}" if page else source


class Retriever:
    """Fetch the chunks most relevant to a question.

    Wraps an :class:`~embeddings.Embedder` and a
    :class:`~vectorstore.VectorStore` that were built with the same
    embedding model — querying with a different model than the index
    was built with returns plausible-looking nonsense, so the embedder
    is a required constructor argument, never a hidden default.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        k: int = DEFAULT_K,
        min_score: float | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.store = store
        self.embedder = embedder
        self.k = k
        self.min_score = min_score

    def retrieve(
        self, query: str, k: int | None = None
    ) -> list[RetrievedChunk]:
        """Return the top-``k`` chunks for one question, best first.

        ``min_score`` (if set at construction) drops hits below the
        threshold, so an off-topic question can return an empty list
        instead of the least-irrelevant chunks in the store.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        query_vector = self.embedder.embed([query])[0]
        return self._wrap(self.store.search(query_vector, k=k or self.k))

    def retrieve_batch(
        self, queries: list[str], k: int | None = None
    ) -> list[list[RetrievedChunk]]:
        """Retrieve for several questions in one embedding pass."""
        if not queries:
            return []
        vectors = self.embedder.embed(queries)
        return [
            self._wrap(self.store.search(vector, k=k or self.k))
            for vector in vectors
        ]

    def _wrap(self, hits: list[SearchHit]) -> list[RetrievedChunk]:
        wrapped = [
            RetrievedChunk(chunk=hit.chunk, score=hit.score) for hit in hits
        ]
        if self.min_score is not None:
            wrapped = [w for w in wrapped if w.score >= self.min_score]
        return wrapped


def format_context(hits: list[RetrievedChunk]) -> str:
    """Render hits as a numbered context block for a prompt.

    Each passage is headed by its citation in brackets —
    ``[1] PMID:38291023`` — matching the citation style ``prompts.py``
    asks the model to use in its answer.
    """
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] {hit.citation}\n{hit.text}")
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve the chunks most relevant to a question."
    )
    parser.add_argument("store", help="Vector store stem (from vectorstore.py)")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--model", default=None,
                        help="Embedding model (default: embeddings.DEFAULT_MODEL)")
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    store = VectorStore.load(args.store)
    embedder = Embedder(model_name=args.model) if args.model else Embedder()
    retriever = Retriever(store, embedder, k=args.k, min_score=args.min_score)
    hits = retriever.retrieve(args.query)
    if not hits:
        print("no chunks retrieved (empty store or below --min-score)")
        return
    print(format_context(hits))


if __name__ == "__main__":
    main()
