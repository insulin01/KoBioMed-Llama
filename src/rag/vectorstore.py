"""FAISS vector store for the KoBioMed-Llama RAG pipeline.

Indexes the embedding matrix produced by ``embeddings.py`` and answers
nearest-neighbour queries with the chunk each vector came from, so a
search result is never just an id — it is the full chunk record
(text + provenance from ``chunking.py``), ready to cite.

Design notes:

- The index is ``IndexFlatIP`` (exact inner-product search). Because
  ``embeddings.py`` L2-normalizes vectors by default, inner product is
  cosine similarity. Exact search is the right default at this
  project's scale (tens of thousands of chunks); approximate indexes
  (IVF/HNSW) can replace the factory string later without changing
  this module's interface.
- Ids are the chunk ids from ``chunking.py``. The store keeps the id
  list and the chunk dicts in row order alongside the FAISS index, and
  :meth:`VectorStore.save` / :meth:`VectorStore.load` persist all
  three together so rows can never drift out of alignment.

Requires ``faiss-cpu`` (see requirements.txt). The import is deferred
to first index use so this module's helpers stay importable without it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _import_faiss():
    """Import faiss with an actionable error if it is missing."""
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss is required for the vector store. "
            "Install it with: pip install faiss-cpu"
        ) from exc
    return faiss


@dataclass
class SearchHit:
    """One search result: the chunk and its similarity score."""

    chunk: dict
    score: float

    @property
    def chunk_id(self) -> str:
        return str(self.chunk.get("chunk_id", ""))


class VectorStore:
    """An exact-search FAISS index over chunk embeddings.

    Rows, ids, and chunk dicts are kept in lockstep: row ``i`` of the
    index is ``ids[i]`` and ``chunks[i]``. All mutations go through
    :meth:`add`, which validates alignment before touching the index.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        faiss = _import_faiss()
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self.ids: list[str] = []
        self.chunks: list[dict] = []

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def from_embeddings(
        cls, embeddings: np.ndarray, chunks: list[dict]
    ) -> "VectorStore":
        """Build a store from an embedding matrix and its chunk dicts.

        *chunks* must carry a ``chunk_id`` each (as produced by
        ``chunking.py``) and line up row-for-row with *embeddings*.
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D matrix (n, dim)")
        store = cls(dimension=int(embeddings.shape[1]))
        store.add(embeddings, chunks)
        return store

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """Append rows to the index.

        Vectors are expected to be L2-normalized already (the default
        output of ``embeddings.Embedder``); scores are inner products.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"embeddings must have shape (n, {self.dimension}), "
                f"got {embeddings.shape}"
            )
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"row count ({embeddings.shape[0]}) must match "
                f"chunk count ({len(chunks)})"
            )
        new_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
        if any(not cid for cid in new_ids):
            raise ValueError("every chunk must have a non-empty chunk_id")
        if len(set(new_ids)) != len(new_ids) or set(new_ids) & set(self.ids):
            raise ValueError("chunk_id values must be unique within the store")
        self._index.add(embeddings)
        self.ids.extend(new_ids)
        self.chunks.extend(chunks)

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchHit]:
        """Return the *k* chunks most similar to *query*.

        *query* is a single embedding vector (shape ``(dim,)`` or
        ``(1, dim)``). Results are ordered by descending score; fewer
        than *k* hits come back when the store holds fewer chunks.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if not self.ids:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"query dim ({query.shape[1]}) does not match "
                f"index dim ({self.dimension})"
            )
        scores, rows = self._index.search(query, min(k, len(self.ids)))
        return [
            SearchHit(chunk=self.chunks[row], score=float(score))
            for score, row in zip(scores[0], rows[0])
            if row >= 0
        ]

    def save(self, path: str | Path) -> Path:
        """Persist index + ids + chunks under *path*.

        Writes ``<path>.faiss`` (the FAISS index), ``<path>.ids.json``
        (row ids in order), and ``<path>.chunks.json`` (chunk dicts in
        row order). :meth:`load` reads the same trio back.
        """
        faiss = _import_faiss()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        with path.with_suffix(".ids.json").open("w", encoding="utf-8") as f:
            json.dump(self.ids, f, ensure_ascii=False, indent=1)
        with path.with_suffix(".chunks.json").open("w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Load a store written by :meth:`save`, verifying alignment."""
        faiss = _import_faiss()
        path = Path(path)
        index = faiss.read_index(str(path.with_suffix(".faiss")))
        with path.with_suffix(".ids.json").open(encoding="utf-8") as f:
            ids = json.load(f)
        with path.with_suffix(".chunks.json").open(encoding="utf-8") as f:
            chunks = json.load(f)
        if not (len(ids) == len(chunks) == index.ntotal):
            raise ValueError(
                f"saved store is misaligned: {len(ids)} ids, "
                f"{len(chunks)} chunks, {index.ntotal} index rows"
            )
        store = cls.__new__(cls)
        store.dimension = int(index.d)
        store._index = index
        store.ids = ids
        store.chunks = chunks
        return store


def build_from_files(
    chunks_path: str | Path, embeddings_stem: str | Path
) -> VectorStore:
    """Build a store from a chunks JSONL and an embeddings pair.

    *embeddings_stem* is the ``--out`` stem used by ``embeddings.py``
    (``<stem>.npy`` + ``<stem>.ids.json``). Chunk order in the JSONL
    must match the embedding id order; a mismatch raises ``ValueError``
    instead of silently misaligning text and vectors.
    """
    from embeddings import load_embeddings  # local module, deferred

    with Path(chunks_path).open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    matrix, emb_ids = load_embeddings(embeddings_stem)
    by_id = {str(c.get("chunk_id", "")): c for c in chunks}
    if set(by_id) != set(emb_ids):
        raise ValueError("chunk ids and embedding ids do not match")
    ordered = [by_id[cid] for cid in emb_ids]
    return VectorStore.from_embeddings(matrix, ordered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a FAISS vector store from chunks + embeddings."
    )
    parser.add_argument("chunks", help="Chunks JSONL from chunking.py")
    parser.add_argument("embeddings", help="Embeddings stem from embeddings.py")
    parser.add_argument("--out", default="data/vectorstore",
                        help="Output stem: <out>.faiss + .ids.json + .chunks.json")
    args = parser.parse_args()

    store = build_from_files(args.chunks, args.embeddings)
    store.save(args.out)
    print(f"indexed {len(store)} chunk(s), dim={store.dimension} -> {args.out}.faiss")


if __name__ == "__main__":
    main()
