"""Text chunking for the KoBioMed-Llama RAG pipeline.

Splits long documents (PubMed abstracts from ``pubmed.py``, page text from
``pdf_ingest.py``) into overlapping chunks suitable for embedding and
retrieval. Splits prefer sentence boundaries so a chunk never cuts a
sentence in half unless the sentence itself exceeds the chunk size.

Every chunk keeps its provenance — source document, page number, PMID —
so retrieved chunks can always be cited back to the original text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

#: Default chunk size in characters (~250 tokens for English text).
DEFAULT_CHUNK_SIZE = 1000

#: Default overlap between consecutive chunks, in characters.
DEFAULT_OVERLAP = 200

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    """One retrievable text chunk with its provenance."""

    chunk_id: str
    text: str
    source: str
    page: int | None = None
    index: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _chunk_id(source: str, page: int | None, index: int, text: str) -> str:
    """Stable short id derived from provenance + content."""
    key = f"{source}|{page}|{index}|{text[:64]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences (heuristic, biomedical-friendly).

    Keeps abbreviations like "et al." and decimal numbers ("0.05") from
    being treated as sentence ends by requiring the next sentence to start
    with a capital letter or digit after terminal punctuation.
    """
    text = " ".join(text.split())
    if not text:
        return []
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split(sentence: str, chunk_size: int) -> list[str]:
    """Split one oversized sentence by character window."""
    return [
        sentence[i : i + chunk_size].strip()
        for i in range(0, len(sentence), chunk_size)
        if sentence[i : i + chunk_size].strip()
    ]


def chunk_text(
    text: str,
    source: str,
    page: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split *text* into sentence-aware overlapping chunks.

    Sentences are greedily packed until adding another would exceed
    *chunk_size*; the tail sentences of each chunk (up to *overlap*
    characters) seed the next chunk so context is not lost at boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    sentences: list[str] = []
    for sent in split_sentences(text):
        if len(sent) > chunk_size:
            sentences.extend(_hard_split(sent, chunk_size))
        else:
            sentences.append(sent)

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        body = " ".join(current)
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(source, page, idx, body),
                text=body,
                source=source,
                page=page,
                index=idx,
                metadata=dict(metadata or {}),
            )
        )
        # Keep the tail (up to `overlap` chars) as the seed of the next chunk.
        tail: list[str] = []
        tail_len = 0
        for sent in reversed(current):
            if tail_len + len(sent) + 1 > overlap:
                break
            tail.insert(0, sent)
            tail_len += len(sent) + 1
        current, current_len = tail, tail_len

    for sent in sentences:
        if current and current_len + len(sent) + 1 > chunk_size:
            flush()
        current.append(sent)
        current_len += len(sent) + 1
    # Final flush without seeding a next chunk.
    if current:
        body = " ".join(current)
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(source, page, idx, body),
                text=body,
                source=source,
                page=page,
                index=idx,
                metadata=dict(metadata or {}),
            )
        )
    return chunks


def chunk_pdf_document(
    document: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk one document dict produced by ``pdf_ingest.save_jsonl``.

    Pages are chunked independently so every chunk carries an exact page.
    """
    source = document.get("source", "unknown.pdf")
    chunks: list[Chunk] = []
    for page in document.get("pages", []):
        page_chunks = chunk_text(
            page.get("text", ""),
            source=source,
            page=page.get("page"),
            chunk_size=chunk_size,
            overlap=overlap,
            metadata={"title": document.get("title", "")},
        )
        chunks.extend(page_chunks)
    # Re-index across the whole document.
    for i, chunk in enumerate(chunks):
        chunk.index = i
        chunk.chunk_id = _chunk_id(chunk.source, chunk.page, i, chunk.text)
    return chunks


def chunk_pubmed_record(
    record: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk one record dict produced by ``pubmed.save_jsonl``.

    Title and abstract are chunked together; metadata keeps PMID/DOI so
    retrieval results can cite the article.
    """
    text = " ".join(
        part for part in [record.get("title", ""), record.get("abstract", "")] if part
    )
    return chunk_text(
        text,
        source=f"PMID:{record.get('pmid', '')}",
        chunk_size=chunk_size,
        overlap=overlap,
        metadata={
            "pmid": record.get("pmid", ""),
            "doi": record.get("doi", ""),
            "journal": record.get("journal", ""),
            "year": record.get("year", ""),
        },
    )


def save_jsonl(chunks: Iterable[Chunk], path: str | Path) -> Path:
    """Write chunks as JSONL (one JSON object per chunk)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file back into a list of dicts."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk a documents JSONL (from pubmed.py or pdf_ingest.py) "
        "into retrieval-ready chunks."
    )
    parser.add_argument("input", help="Input JSONL of documents/records")
    parser.add_argument("--kind", choices=["pdf", "pubmed"], required=True)
    parser.add_argument("--out", default="data/chunks.jsonl")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()

    chunker = chunk_pdf_document if args.kind == "pdf" else chunk_pubmed_record
    all_chunks: list[Chunk] = []
    for doc in load_jsonl(args.input):
        all_chunks.extend(
            chunker(doc, chunk_size=args.chunk_size, overlap=args.overlap)
        )
    out = save_jsonl(all_chunks, args.out)
    print(f"wrote {len(all_chunks)} chunk(s) -> {out}")


if __name__ == "__main__":
    main()
