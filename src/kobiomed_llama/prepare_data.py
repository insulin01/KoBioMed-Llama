"""Training-data preparation for KoBioMed-Llama.

Turns the outputs of the RAG pipeline into instruction-tuning JSONL
that ``train_lora.py`` can consume directly. The script *assembles and
validates* data — it never invents answers. Every example it writes
comes from one of two honest sources:

1. **Curated seed Q&A** (``--seed-qa``): question/answer pairs written
   by a researcher (or produced by a reviewed LLM run), each
   optionally carrying the retrieved context its answer cites. These
   are the primary, high-quality examples.
2. **Extractive summaries from chunks** (``--chunks``): for each chunk
   produced by ``chunking.py``, a summarization example whose target
   output is an *extractive* summary — the chunk's own most salient
   sentences, selected by a transparent scoring rule. No text is
   generated or paraphrased, so the output can never hallucinate a
   finding the source does not contain.

Output records use the Alpaca-style shape
``prompts.format_instruction_example`` defines (``system`` /
``instruction`` / ``input`` / ``output``), plus a ``source`` field for
provenance. The dataset is deduplicated, shuffled with a fixed seed,
split into train/validation, and every record is validated before it
is written: non-empty fields, length caps, and — for answers that
cite passages — citations that actually resolve against the provided
context (``prompts.validate_citations``).

Example usage::

    python prepare_data.py \
        --chunks data/chunks.jsonl \
        --seed-qa data/seed_qa.jsonl \
        --out-train data/train.jsonl \
        --out-val data/val.jsonl \
        --val-fraction 0.1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

try:
    from prompts import (
        INSTRUCTION_SYSTEM_PROMPT,
        format_instruction_example,
        validate_citations,
    )
except ImportError:  # when imported as part of the kobiomed_llama package
    from .prompts import (  # type: ignore[no-redef]
        INSTRUCTION_SYSTEM_PROMPT,
        format_instruction_example,
        validate_citations,
    )

#: Default fraction of examples held out for validation.
DEFAULT_VAL_FRACTION = 0.1

#: Fixed shuffle seed so a re-run over the same inputs reproduces the
#: same train/val split.
DEFAULT_SEED = 42

#: Maximum characters kept in an example's ``input`` (context). Chunks
#: from ``chunking.py`` are ~1,000 chars, so this cap only bites on
#: unusually long seed contexts.
MAX_INPUT_CHARS = 6_000

#: Maximum characters allowed in an example's ``output``.
MAX_OUTPUT_CHARS = 4_000

#: Maximum characters allowed in an ``instruction``.
MAX_INSTRUCTION_CHARS = 1_000

#: Sentences kept in an extractive summary (see ``extractive_summary``).
DEFAULT_SUMMARY_SENTENCES = 2

#: Chunks shorter than this are skipped for summary examples — there is
#: nothing to summarize.
MIN_CHUNK_CHARS = 200

#: Instruction text used for chunk-derived summarization examples.
SUMMARIZE_INSTRUCTION = (
    "다음 생의학 문헌 구절을 한국어로 핵심만 간추려 1-2문장으로 요약하세요. "
    "수치(효과 크기, p값, 표본 수)가 있으면 그대로 보존하고, "
    "구절에 없는 내용을 덧붙이지 마세요."
)

#: Instruction text used for context-grounded QA examples built from
#: seed pairs that carry retrieved context.
SEED_QA_INSTRUCTION_SUFFIX = (
    "\n\n다음 <context>의 문헌 구절만을 근거로 답하세요. "
    "사실적 주장마다 해당 구절 번호를 [n] 형태로 인용하고, "
    "구절에 답이 없으면 \"제공된 문헌에서는 확인할 수 없습니다\"라고 답하세요."
)

#: Regexes used by the extractive-summary scorer.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9가-힣])")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|배|fold|mg|mL|n\s*=)?", re.IGNORECASE)
_P_VALUE_RE = re.compile(r"p\s*[<=>]\s*0\.\d+", re.IGNORECASE)
_GENE_LIKE_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)?\b")
_WORD_RE = re.compile(r"[A-Za-z가-힣]{2,}")

#: Words that carry no topical signal for the summarizer.
_STOPWORDS = frozenset(
    {
        "the", "and", "with", "from", "that", "this", "were", "have",
        "been", "also", "such", "than", "into", "over", "more", "when",
        "which", "their", "there", "these", "those", "study", "studies",
        "results", "showed", "using", "used", "cells", "cell",
        "그러나", "그리고", "또한", "이는", "이를", "에서", "으로",
        "대한", "있는", "연구", "결과", "통해", "경우",
    }
)


@dataclass
class DatasetStats:
    """Counts collected while building a dataset."""

    seed_qa_in: int = 0
    chunks_in: int = 0
    examples_built: int = 0
    dropped: Counter = field(default_factory=Counter)
    duplicates_removed: int = 0
    train_examples: int = 0
    val_examples: int = 0

    def report(self) -> str:
        lines = [
            f"seed Q&A records read: {self.seed_qa_in}",
            f"chunks read: {self.chunks_in}",
            f"examples built: {self.examples_built}",
            f"duplicates removed: {self.duplicates_removed}",
            f"train examples: {self.train_examples}",
            f"validation examples: {self.val_examples}",
        ]
        if self.dropped:
            lines.append("dropped:")
            for reason, count in self.dropped.most_common():
                lines.append(f"  {reason}: {count}")
        return "\n".join(lines)


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts (blank lines skipped)."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: Iterable[dict], path: str | Path) -> Path:
    """Write records as JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (same heuristic as ``chunking.py``)."""
    text = " ".join(text.split())
    if not text:
        return []
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _sentence_score(sentence: str, word_freq: Counter) -> float:
    """Salience score for one sentence of a chunk.

    Rewards sentences that carry checkable content — numbers, p-values,
    gene/protein-like tokens, and the chunk's frequent content words —
    because those are the sentences a summary must preserve. The score
    is a heuristic for *selection*, not a quality judgement.
    """
    score = 0.0
    score += 2.0 * len(_NUMBER_RE.findall(sentence))
    score += 3.0 * len(_P_VALUE_RE.findall(sentence))
    score += 1.5 * len(_GENE_LIKE_RE.findall(sentence))
    words = [w.lower() for w in _WORD_RE.findall(sentence)]
    score += sum(
        word_freq[w] for w in words if w not in _STOPWORDS
    ) / max(len(words), 1)
    return score


def extractive_summary(text: str, n_sentences: int = DEFAULT_SUMMARY_SENTENCES) -> str:
    """Return an extractive summary: the chunk's top sentences.

    The top ``n_sentences`` by salience score are returned **in their
    original order**, so the summary reads in the source's own flow.
    The output is always a verbatim subset of the input — nothing is
    paraphrased or added.
    """
    sentences = split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= n_sentences:
        return " ".join(sentences)
    words = [w.lower() for w in _WORD_RE.findall(text)]
    word_freq = Counter(w for w in words if w not in _STOPWORDS)
    scored = sorted(
        enumerate(sentences),
        key=lambda pair: _sentence_score(pair[1], word_freq),
        reverse=True,
    )
    chosen = sorted(index for index, _ in scored[:n_sentences])
    return " ".join(sentences[i] for i in chosen)


def _chunk_label(chunk: dict) -> str:
    """Human-readable source label for a chunk (for provenance)."""
    metadata = chunk.get("metadata", {}) or {}
    pmid = metadata.get("pmid") or ""
    if pmid:
        return f"PMID:{pmid}"
    source = str(chunk.get("source", "unknown"))
    page = chunk.get("page")
    return f"{source}, p. {page}" if page else source


def build_summary_examples(
    chunks: Sequence[dict],
    n_sentences: int = DEFAULT_SUMMARY_SENTENCES,
    stats: DatasetStats | None = None,
) -> list[dict]:
    """Build one summarization example per chunk.

    Instruction: summarize the passage in Korean. Input: the chunk
    text. Output: the extractive summary. Chunks too short to
    summarize, or whose summary would just repeat the whole chunk,
    are skipped (counted in ``stats``).
    """
    examples: list[dict] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if len(text) < MIN_CHUNK_CHARS:
            if stats is not None:
                stats.dropped["chunk too short for summary"] += 1
            continue
        summary = extractive_summary(text, n_sentences=n_sentences)
        if not summary or summary == text:
            if stats is not None:
                stats.dropped["summary equals full chunk"] += 1
            continue
        examples.append(
            {
                **format_instruction_example(
                    instruction=SUMMARIZE_INSTRUCTION,
                    output=summary,
                    input_text=text,
                ),
                "source": _chunk_label(chunk),
                "kind": "summary",
            }
        )
    if stats is not None:
        stats.examples_built += len(examples)
    return examples


def build_qa_examples(
    seed_records: Sequence[dict],
    stats: DatasetStats | None = None,
) -> list[dict]:
    """Convert curated seed Q&A records into instruction examples.

    Seed record shape::

        {"question": "...", "answer": "...",
         "context": "<optional numbered passages>",
         "n_context_passages": <optional int>,
         "source": "<optional provenance label>"}

    When ``context`` is present, the instruction gains the
    cite-from-context suffix and the answer's citations are checked
    against ``n_context_passages`` (or against the number of ``[n]``
    headings found in the context when the count is not given).
    Records failing validation are skipped and counted in ``stats``.
    """
    examples: list[dict] = []
    for record in seed_records:
        question = str(record.get("question", "")).strip()
        answer = str(record.get("answer", "")).strip()
        context = str(record.get("context", "") or "").strip()
        if not question or not answer:
            if stats is not None:
                stats.dropped["seed record missing question/answer"] += 1
            continue
        if context:
            n_passages = record.get("n_context_passages")
            if n_passages is None:
                n_passages = len(re.findall(r"^\[\d+\]", context, re.MULTILINE))
            problems = validate_citations(answer, int(n_passages))
            if problems:
                if stats is not None:
                    stats.dropped[
                        "seed answer citations do not resolve: "
                        + "; ".join(problems)
                    ] += 1
                continue
            instruction = question + SEED_QA_INSTRUCTION_SUFFIX
            input_text = context
        else:
            instruction = question
            input_text = ""
        examples.append(
            {
                **format_instruction_example(
                    instruction=instruction,
                    output=answer,
                    input_text=input_text,
                ),
                "source": str(record.get("source", "seed")),
                "kind": "qa",
            }
        )
    if stats is not None:
        stats.examples_built += len(examples)
    return examples


def _example_key(example: dict) -> str:
    """Stable dedup key: hash of instruction + input."""
    raw = f"{example.get('instruction', '')}\n{example.get('input', '')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def deduplicate(
    examples: list[dict], stats: DatasetStats | None = None
) -> list[dict]:
    """Drop exact duplicates (same instruction + input), keeping the first."""
    seen: set[str] = set()
    kept: list[dict] = []
    for example in examples:
        key = _example_key(example)
        if key in seen:
            if stats is not None:
                stats.duplicates_removed += 1
            continue
        seen.add(key)
        kept.append(example)
    return kept


def validate_example(example: dict) -> list[str]:
    """Check one built example; returns a list of problems (empty = OK)."""
    problems: list[str] = []
    for field_name in ("system", "instruction", "output"):
        if not str(example.get(field_name, "")).strip():
            problems.append(f"empty {field_name}")
    if len(str(example.get("instruction", ""))) > MAX_INSTRUCTION_CHARS:
        problems.append("instruction over length cap")
    if len(str(example.get("input", ""))) > MAX_INPUT_CHARS:
        problems.append("input over length cap")
    if len(str(example.get("output", ""))) > MAX_OUTPUT_CHARS:
        problems.append("output over length cap")
    return problems


def split_train_val(
    examples: list[dict],
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], list[dict]]:
    """Shuffle (fixed seed) and split into train/validation lists."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")
    shuffled = list(examples)
    Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    # Keep at least one validation example when there is anything to split.
    if shuffled and val_fraction > 0 and n_val == 0:
        n_val = 1
    return shuffled[n_val:], shuffled[:n_val]


def build_dataset(
    chunks: Sequence[dict] | None = None,
    seed_qa: Sequence[dict] | None = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    n_summary_sentences: int = DEFAULT_SUMMARY_SENTENCES,
) -> tuple[list[dict], list[dict], DatasetStats]:
    """Build the full dataset from chunks and/or seed Q&A.

    Returns ``(train_examples, val_examples, stats)``. Raises
    ``ValueError`` if no examples survive validation — writing an
    empty training file would silently train on nothing.
    """
    stats = DatasetStats(
        seed_qa_in=len(seed_qa or []),
        chunks_in=len(chunks or []),
    )
    examples: list[dict] = []
    if seed_qa:
        examples.extend(build_qa_examples(seed_qa, stats=stats))
    if chunks:
        examples.extend(
            build_summary_examples(
                chunks, n_sentences=n_summary_sentences, stats=stats
            )
        )

    valid: list[dict] = []
    for example in examples:
        problems = validate_example(example)
        if problems:
            stats.dropped["failed final validation: " + "; ".join(problems)] += 1
            continue
        valid.append(example)
    valid = deduplicate(valid, stats=stats)

    if not valid:
        raise ValueError(
            "no usable examples: check --chunks / --seed-qa inputs "
            f"(dropped: {dict(stats.dropped)})"
        )
    train, val = split_train_val(valid, val_fraction=val_fraction, seed=seed)
    stats.train_examples = len(train)
    stats.val_examples = len(val)
    return train, val, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build KoBioMed-Llama instruction-tuning JSONL "
        "from RAG chunks and curated seed Q&A."
    )
    parser.add_argument("--chunks", default=None,
                        help="Chunks JSONL from chunking.py")
    parser.add_argument("--seed-qa", default=None,
                        help="Curated seed Q&A JSONL "
                        "(question/answer/context/source)")
    parser.add_argument("--out-train", default="data/train.jsonl")
    parser.add_argument("--out-val", default="data/val.jsonl")
    parser.add_argument("--val-fraction", type=float,
                        default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Shuffle seed (default: 42)")
    parser.add_argument("--summary-sentences", type=int,
                        default=DEFAULT_SUMMARY_SENTENCES,
                        help="Sentences per extractive summary")
    parser.add_argument("--no-val", action="store_true",
                        help="Write all examples to train (no val split)")
    parser.add_argument("--print-stats", action="store_true",
                        help="Print the build report (default: always "
                        "printed to stderr)")
    args = parser.parse_args()

    if args.chunks is None and args.seed_qa is None:
        parser.error("at least one of --chunks or --seed-qa is required")

    chunks = load_jsonl(args.chunks) if args.chunks else None
    seed_qa = load_jsonl(args.seed_qa) if args.seed_qa else None
    train, val, stats = build_dataset(
        chunks=chunks,
        seed_qa=seed_qa,
        val_fraction=0.0 if args.no_val else args.val_fraction,
        seed=args.seed,
        n_summary_sentences=args.summary_sentences,
    )
    train_path = write_jsonl(train, args.out_train)
    val_path = write_jsonl(val, args.out_val) if val else None
    print(stats.report(), file=sys.stderr)
    print(f"wrote {len(train)} train example(s) -> {train_path}")
    if val_path is not None:
        print(f"wrote {len(val)} validation example(s) -> {val_path}")


if __name__ == "__main__":
    main()
