"""Text embeddings for the KoBioMed-Llama RAG pipeline.

Wraps ``sentence-transformers`` so the rest of the pipeline (the FAISS
vector store in ``vectorstore.py``) never touches the model directly:
give it chunk texts, get back a normalized ``numpy`` matrix, one row per
chunk, ready to index.

Embeddings are L2-normalized by default, which makes cosine similarity
equivalent to a plain inner product — the metric the vector store will
use. Batching and progress reporting are handled here, so callers just
pass a list of texts.

Model choice: the default ``all-MiniLM-L6-v2`` is small (80 MB, 384
dimensions) and fast on CPU, a good baseline for the pipeline. For
biomedical text, swap in a domain model such as
``pritamdeka/S-PubMedBert-MS-MARCO`` via ``model_name=`` — the interface
is unchanged.

Requires ``sentence-transformers`` (see requirements.txt). The import is
deferred to ``Embedder`` construction so the pure helpers in this module
(saving/loading embedding matrices) work without the dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

#: Default embedding model: small, fast on CPU, 384-dimensional output.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Default batch size for encoding (tuned for CPU inference).
DEFAULT_BATCH_SIZE = 32


class Embedder:
    """A sentence-transformers model with a pipeline-friendly interface.

    The model is loaded lazily on first use (not at construction) so a
    script can parse arguments and validate inputs before paying the
    load cost, and so multiple ``Embedder`` objects can share one model
    via :meth:`from_shared`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        """Create an embedder.

        Args:
            model_name: Hugging Face model id (or local path).
            batch_size: Texts encoded per model call.
            normalize: L2-normalize each vector (recommended: enables
                inner-product search to act as cosine similarity).
            device: ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self._model = None

    @property
    def model(self):  # type: ignore[no-untyped-def]
        """Load the underlying SentenceTransformer on first access."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for embeddings. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension of the loaded model."""
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* into a float32 matrix of shape (n, dim).

        Empty strings are encoded as-is by the model (they still get a
        row, so output rows always line up with input texts). Raises
        ``ValueError`` on an empty input list.
        """
        if not texts:
            raise ValueError("texts must not be empty")
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) >= self.batch_size,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_chunks(self, chunks: Iterable[dict]) -> tuple[list[dict], np.ndarray]:
        """Embed the ``text`` field of chunk dicts (from ``chunking.py``).

        Returns the chunks (unchanged, in input order) alongside the
        embedding matrix, so ids and vectors stay aligned. Chunks with
        an empty text still get a row — use the returned matrix rows in
        order, never by re-indexing on text.
        """
        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("chunks must not be empty")
        texts = [str(chunk.get("text", "")) for chunk in chunk_list]
        return chunk_list, self.embed(texts)


def save_embeddings(
    embeddings: np.ndarray, ids: list[str], path: str | Path
) -> Path:
    """Save an embedding matrix plus its row ids.

    Writes ``<path>.npy`` (the matrix) and ``<path>.ids.json`` (the id
    list, in row order) next to each other, so a later load can check
    that ids and rows still line up. ``vectorstore.py`` reads this pair.
    """
    path = Path(path)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2-D matrix (n, dim)")
    if len(ids) != embeddings.shape[0]:
        raise ValueError(
            f"ids ({len(ids)}) must match embedding rows ({embeddings.shape[0]})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), embeddings.astype(np.float32))
    with path.with_suffix(".ids.json").open("w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=1)
    return path


def load_embeddings(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load an embedding matrix and its row ids written by :func:`save_embeddings`."""
    path = Path(path)
    matrix = np.load(path.with_suffix(".npy"))
    with path.with_suffix(".ids.json").open(encoding="utf-8") as f:
        ids = json.load(f)
    if len(ids) != matrix.shape[0]:
        raise ValueError(
            f"id count ({len(ids)}) does not match matrix rows ({matrix.shape[0]})"
        )
    return matrix, ids


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against every matrix row.

    Works for normalized and unnormalized inputs alike (each vector is
    normalized here); with the default normalized embeddings this is
    just the inner product. Returns a 1-D array of length ``matrix.shape[0]``.
    """
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    if query.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"query dim ({query.shape[0]}) does not match matrix dim ({matrix.shape[1]})"
        )
    q_norm = query / (np.linalg.norm(query) or 1.0)
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    return (matrix / row_norms) @ q_norm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed chunk texts (from chunking.py) and save the matrix."
    )
    parser.add_argument("input", help="Input chunks JSONL from chunking.py")
    parser.add_argument("--out", default="data/embeddings",
                        help="Output stem: <out>.npy + <out>.ids.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip L2 normalization of output vectors")
    args = parser.parse_args()

    with Path(args.input).open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    if not chunks:
        raise ValueError(f"no chunks found in {args.input}")

    embedder = Embedder(
        model_name=args.model,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
    )
    _, matrix = embedder.embed_chunks(chunks)
    ids = [chunk.get("chunk_id", str(i)) for i, chunk in enumerate(chunks)]
    save_embeddings(matrix, ids, args.out)
    print(f"embedded {matrix.shape[0]} chunk(s) x {matrix.shape[1]} dims -> {args.out}.npy")


if __name__ == "__main__":
    main()
