"""PDF text extraction for the KoBioMed-Llama RAG pipeline.

Reads a research-paper PDF page by page with ``pypdf``, cleans the raw
extracted text (hyphenated line breaks, stray whitespace), and stores the
result as JSONL — one record per document, with per-page text preserved so
the downstream chunker (``src/rag/chunking.py``) can keep source + page
metadata attached to every chunk.

Install the only runtime dependency with::

    pip install pypdf
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "pypdf is required for PDF ingestion. Install it with "
        "'pip install pypdf'."
    ) from exc

#: Pages whose cleaned text is shorter than this are dropped from output.
#: Extracted headers/footers and blank pages typically fall below it.
MIN_PAGE_CHARS = 20


@dataclass
class PdfPage:
    """Text of a single PDF page (1-based page number)."""

    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PdfDocument:
    """A whole PDF: metadata plus one record per page."""

    source: str
    num_pages: int
    title: str = ""
    pages: list[PdfPage] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        return data

    @property
    def full_text(self) -> str:
        """All page texts joined in reading order."""
        return "\n\n".join(p.text for p in self.pages)


def clean_text(text: str) -> str:
    """Clean raw text extracted from a PDF page.

    - Rejoins words split by a hyphen at a line break (``mitochon-\\n`` +
      ``dria`` -> ``mitochondria``).
    - Collapses runs of spaces/tabs and more than two newlines.
    - Strips leading/trailing whitespace on every line.

    Paragraph breaks (single newlines between non-empty lines) are kept
    so the chunker can still see sentence boundaries.
    """
    if not text:
        return ""
    # Rejoin hyphenated line breaks: "mitochon-\ndria" -> "mitochondria".
    text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)
    # A line break that is clearly a wrapped line (lowercase continuation)
    # becomes a space; breaks after sentence-ending punctuation are kept.
    text = re.sub(r"([a-z,;:])\n([a-z])", r"\1 \2", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_pages(
    pdf_path: str | Path,
    clean: bool = True,
    min_chars: int = MIN_PAGE_CHARS,
) -> list[PdfPage]:
    """Extract text from every page of *pdf_path*.

    Returns pages in order, dropping pages that yield almost no text
    (blank pages, image-only scans). Encrypted PDFs are retried with an
    empty password, which covers the common "open with no password" case;
    a truly password-protected file raises ``ValueError``.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"PDF is encrypted and cannot be opened: {path}") from exc
    pages: list[PdfPage] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if clean:
            text = clean_text(text)
        if len(text) < min_chars:
            continue
        pages.append(PdfPage(page=i, text=text))
    return pages


def extract_document(
    pdf_path: str | Path,
    clean: bool = True,
    min_chars: int = MIN_PAGE_CHARS,
) -> PdfDocument:
    """Extract a whole PDF into a :class:`PdfDocument` with metadata."""
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    title = ""
    try:
        meta = reader.metadata
        if meta is not None:
            title = str(meta.title or "")
    except Exception:
        title = ""
    pages = extract_pages(path, clean=clean, min_chars=min_chars)
    return PdfDocument(
        source=path.name,
        num_pages=len(reader.pages),
        title=title,
        pages=pages,
    )


def iter_pdfs(directory: str | Path) -> Iterator[Path]:
    """Yield ``*.pdf`` files under *directory*, sorted by name."""
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    yield from sorted(path.rglob("*.pdf"))


def save_jsonl(documents: Iterable[PdfDocument], path: str | Path) -> Path:
    """Write documents as JSONL (one JSON object per document)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file back into a list of dicts."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ingest_directory(
    directory: str | Path,
    out_path: str | Path = "data/pdf_docs.jsonl",
    clean: bool = True,
) -> tuple[Path, int]:
    """Extract every PDF under *directory* and save them to JSONL.

    Returns ``(output_path, number_of_documents)``. Files that fail to
    parse are skipped with a warning on stderr rather than aborting the
    whole batch, so one corrupt PDF cannot block an ingest run.
    """
    import sys

    documents: list[PdfDocument] = []
    for pdf_path in iter_pdfs(directory):
        try:
            documents.append(extract_document(pdf_path, clean=clean))
        except Exception as exc:  # corrupt/encrypted file: skip, don't abort
            print(f"warning: skipping {pdf_path}: {exc}", file=sys.stderr)
    return save_jsonl(documents, out_path), len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF (or a directory of PDFs) to JSONL."
    )
    parser.add_argument("input", help="PDF file or directory containing PDFs")
    parser.add_argument("--out", default="data/pdf_docs.jsonl")
    parser.add_argument("--no-clean", action="store_true",
                        help="Keep raw extracted text without cleaning")
    args = parser.parse_args()

    clean = not args.no_clean
    in_path = Path(args.input)
    if in_path.is_dir():
        out, count = ingest_directory(in_path, args.out, clean=clean)
        print(f"ingested {count} PDF(s) -> {out}")
    else:
        doc = extract_document(in_path, clean=clean)
        out = save_jsonl([doc], args.out)
        print(f"extracted {len(doc.pages)}/{doc.num_pages} page(s) "
              f"from {doc.source} -> {out}")


if __name__ == "__main__":
    main()
