"""PubMed literature collection via NCBI Entrez.

Searches PubMed for a query, fetches article metadata and abstracts,
and stores them as JSONL — ready for downstream chunking and indexing.

Requires an NCBI-registered email (Entrez usage policy).
Set the ``NCBI_EMAIL`` environment variable or pass ``email=`` explicitly.
An optional ``NCBI_API_KEY`` raises the rate limit.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from Bio import Entrez

#: Seconds to wait between Entrez requests (≤3/s without an API key).
ENTREZ_DELAY = 0.4

#: Max IDs per efetch call (NCBI recommends ≤200 for XML).
EFETCH_BATCH = 200


@dataclass
class PubMedRecord:
    """One PubMed article with the fields the RAG pipeline needs."""

    pmid: str
    title: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    doi: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _configure(email: str | None = None) -> None:
    """Apply Entrez credentials from args or environment."""
    email = email or os.environ.get("NCBI_EMAIL", "")
    if not email:
        raise ValueError(
            "NCBI email is required by the Entrez usage policy. "
            "Pass email= or set the NCBI_EMAIL environment variable."
        )
    Entrez.email = email
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key


def search_pubmed(
    query: str,
    max_results: int = 100,
    email: str | None = None,
    sort: str = "relevance",
) -> list[str]:
    """Return PMIDs matching *query* (best match first)."""
    _configure(email)
    with Entrez.esearch(
        db="pubmed", term=query, retmax=max_results, sort=sort, retmode="xml"
    ) as handle:
        result = Entrez.read(handle)
    time.sleep(ENTREZ_DELAY)
    return list(result["IdList"])


def _parse_article(art: dict) -> PubMedRecord:
    medline = art["MedlineCitation"]
    article = medline["Article"]

    abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
    authors = [
        f"{a.get('ForeName', '')} {a.get('LastName', '')}".strip()
        for a in article.get("AuthorList", [])
        if a.get("LastName")
    ]
    pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
    doi = ""
    for idn in art.get("PubmedData", {}).get("ArticleIdList", []):
        if idn.attributes.get("IdType") == "doi":
            doi = str(idn)

    return PubMedRecord(
        pmid=str(medline["PMID"]),
        title=str(article.get("ArticleTitle", "")),
        abstract=" ".join(str(s) for s in abstract_parts),
        authors=authors,
        journal=str(article.get("Journal", {}).get("Title", "")),
        year=str(pub_date.get("Year", "")),
        doi=doi,
    )


def fetch_records(
    pmids: Iterable[str], email: str | None = None
) -> list[PubMedRecord]:
    """Fetch metadata + abstracts for *pmids* (batched, rate-limited)."""
    _configure(email)
    pmids = list(pmids)
    records: list[PubMedRecord] = []
    for i in range(0, len(pmids), EFETCH_BATCH):
        batch = pmids[i : i + EFETCH_BATCH]
        with Entrez.efetch(db="pubmed", id=",".join(batch), retmode="xml") as handle:
            articles = Entrez.read(handle)
        time.sleep(ENTREZ_DELAY)
        records.extend(_parse_article(art) for art in articles["PubmedArticle"])
    return records


def save_jsonl(records: Iterable[PubMedRecord], path: str | Path) -> Path:
    """Write records as JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file back into a list of dicts."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collect(
    query: str,
    max_results: int = 100,
    out_path: str | Path = "data/pubmed.jsonl",
    email: str | None = None,
) -> Path:
    """End-to-end: search PubMed for *query* and save abstracts to JSONL."""
    pmids = search_pubmed(query, max_results=max_results, email=email)
    records = fetch_records(pmids, email=email)
    return save_jsonl(records, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect PubMed abstracts into a JSONL file."
    )
    parser.add_argument("query", help="PubMed search query, e.g. 'PHF20 senolytic'")
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--out", default="data/pubmed.jsonl")
    parser.add_argument("--email", default=None, help="Overrides NCBI_EMAIL")
    args = parser.parse_args()

    out = collect(args.query, args.max_results, args.out, args.email)
    print(f"saved {args.max_results} max results -> {out}")


if __name__ == "__main__":
    main()
