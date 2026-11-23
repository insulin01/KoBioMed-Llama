"""QA evaluation for KoBioMed-Llama.

Scores model answers against a curated biomedical QA set and reports
how often the model is right, how often it cites the right sources,
and — just as important — how often it admits it does not know.

The script *measures*; it never generates answers. A prediction file
is scored offline, so the same command can grade a fine-tuned
checkpoint, a RAG pipeline, or a baseline without re-running any
model. The eval set is JSONL, one record per line::

    {"question": "Which organelle produces most of a cell's ATP?",
     "reference": "mitochondria",
     "aliases": ["mitochondria", "mitochondrion"],
     "expected_sources": ["Alberts et al., Molecular Biology of the Cell"],
     "answerable": true}

A prediction file has the same questions plus the model's output::

    {"question": "Which organelle produces most of a cell's ATP?",
     "answer": "The mitochondria produce most ATP. [1]",
     "citations": ["Alberts et al., Molecular Biology of the Cell"]}

If ``citations`` is missing, bracketed labels such as
``[PMID:38291023]`` are parsed out of the answer text instead.

Metrics, per question:

* **exact match** — normalized answer equals the reference.
* **token F1** — SQuAD-style word-overlap F1 against the reference.
* **ROUGE-L F1** — longest-common-subsequence overlap, which rewards
  getting the key phrase in the right order rather than a bag of
  words.
* **alias coverage** — fraction of the ``aliases`` (acceptable answer
  strings) that appear in the answer. Biomedical answers are often
  right in several wordings, so a single reference string is too
  strict on its own.
* **citation precision / recall / F1** — the predicted source labels
  against ``expected_sources``. Only computed for items that list
  expected sources.
* **abstention** — for ``answerable: false`` items the correct
  behaviour is to say the literature does not answer the question
  (``prompts.py`` asks for "제공된 문헌에서는 확인할 수 없습니다");
  answering anyway counts as a failure, however fluent it sounds.

An item counts as **correct** when an unanswerable question is met
with an abstention, or an answerable one matches exactly, contains
any of its aliases, or reaches token F1 >= 0.5. Aliases are
alternative wordings of the same answer, so *any* one of them
counts — requiring all of them would ask the model to say the same
thing twice. The rule is deliberately transparent and printed in
the report — a threshold is a judgement call, not a fact, so it is
exposed as ``--correct-f1``.

Example usage::

    python evaluate_qa.py --eval-set evals/sample_qa.jsonl --predictions runs/baseline.jsonl --output runs/report.json
    python evaluate_qa.py --eval-set evals/sample_qa.jsonl --demo
    python evaluate_qa.py --eval-set evals/sample_qa.jsonl --predictions runs/baseline.jsonl --fail-under 0.6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

#: Token F1 at or above this value counts an answerable item correct.
DEFAULT_CORRECT_F1 = 0.5

#: Phrases that count as abstaining instead of answering. Both the
#: English wording and the Korean wording ``prompts.py`` prescribes.
ABSTENTION_PHRASES = (
    "제공된 문헌에서는 확인할 수 없습니다",
    "확인할 수 없습니다",
    "알 수 없습니다",
    "모르겠습니다",
    "cannot confirm",
    "cannot be confirmed",
    "not enough information",
    "insufficient evidence",
    "no evidence in the provided",
    "i don't know",
)

#: Regex for bracketed source labels, e.g. ``[PMID:38291023]``.
_BRACKET_LABEL_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9 .,_:;()/-]{2,})\]")

#: Articles and punctuation stripped when normalizing for comparison.
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_RE = re.compile(r"[^\w\s가-힣]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+(?:[.\-][A-Za-z0-9가-힣]+)*")

#: The three-question smoke-test set used by ``--demo`` and shipped
#: as ``evals/sample_qa.jsonl``. Textbook facts only, so the fixture
#: stays valid without any literature access: one item the stub gets
#: right, one it gets wrong, one it must abstain on.
SAMPLE_EVAL_SET: list[dict] = [
    {
        "id": "demo-01",
        "question": "Which organelle produces most of a cell's ATP?",
        "reference": "mitochondria",
        "aliases": ["mitochondria", "mitochondrion"],
        "expected_sources": ["Alberts et al., Molecular Biology of the Cell"],
        "answerable": True,
    },
    {
        "id": "demo-02",
        "question": (
            "What is the normal fasting blood glucose range "
            "for a non-diabetic adult?"
        ),
        "reference": "70-99 mg/dL",
        "aliases": ["70-99 mg/dL", "70 to 99 mg/dL"],
        "expected_sources": [],
        "answerable": True,
    },
    {
        "id": "demo-03",
        "question": (
            "What was the 2026 overall survival of drug X "
            "in trial Y?"
        ),
        "reference": "",
        "aliases": [],
        "expected_sources": [],
        "answerable": False,
    },
]


@dataclass
class EvalItem:
    """One question in the eval set, with its gold answer."""

    question: str
    reference: str = ""
    aliases: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    answerable: bool = True
    item_id: str = ""

    @classmethod
    def from_record(cls, record: dict, line_no: int) -> "EvalItem":
        """Build from one JSONL record, rejecting malformed rows."""
        question = str(record.get("question", "")).strip()
        if not question:
            raise ValueError(f"line {line_no}: missing 'question'")
        aliases = record.get("aliases") or []
        sources = record.get("expected_sources") or []
        if not isinstance(aliases, list) or not isinstance(sources, list):
            raise ValueError(f"line {line_no}: 'aliases' and "
                             "'expected_sources' must be lists")
        return cls(
            question=question,
            reference=str(record.get("reference", "")).strip(),
            aliases=[str(a) for a in aliases],
            expected_sources=[str(s) for s in sources],
            answerable=bool(record.get("answerable", True)),
            item_id=str(record.get("id", f"line-{line_no}")),
        )


@dataclass
class Prediction:
    """One model answer, matched to an eval question."""

    question: str
    answer: str = ""
    citations: list[str] = field(default_factory=list)

    @classmethod
    def from_record(cls, record: dict, line_no: int) -> "Prediction":
        """Build from one JSONL record; citations fall back to
        bracketed labels parsed from the answer text."""
        question = str(record.get("question", "")).strip()
        if not question:
            raise ValueError(f"line {line_no}: missing 'question'")
        answer = str(record.get("answer", ""))
        citations = record.get("citations")
        if citations is None:
            citations = parse_bracket_labels(answer)
        elif not isinstance(citations, list):
            raise ValueError(f"line {line_no}: 'citations' must be a list")
        return cls(
            question=question,
            answer=answer,
            citations=[str(c) for c in citations],
        )


@dataclass
class ItemResult:
    """Per-question scores, plus the judgement of correct or not."""

    item_id: str
    question: str
    exact_match: float
    token_f1: float
    rouge_l_f1: float
    alias_coverage: float | None
    citation_precision: float | None
    citation_recall: float | None
    citation_f1: float | None
    abstained: bool
    correct: bool


@dataclass
class EvalReport:
    """Aggregate scores over the whole eval set."""

    n_items: int
    n_scored: int
    n_missing_predictions: int
    n_unmatched_predictions: int
    exact_match: float
    token_f1: float
    rouge_l_f1: float
    alias_coverage: float | None
    citation_precision: float | None
    citation_recall: float | None
    citation_f1: float | None
    abstention_accuracy: float | None
    accuracy: float
    items: list[ItemResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Plain dict for JSON output (dataclasses nested)."""
        data = asdict(self)
        return data


def normalize_text(text: str) -> str:
    """Lowercase, strip articles and punctuation, collapse spaces.

    The same normalization SQuAD uses, so "The Mitochondria." and
    "mitochondria" compare equal. Korean text passes through
    unchanged apart from case and punctuation.
    """
    text = text.lower()
    text = _ARTICLES_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Word tokens of the normalized text."""
    return _TOKEN_RE.findall(normalize_text(text))


def exact_match_score(prediction: str, reference: str) -> float:
    """1.0 when the normalized strings are identical, else 0.0."""
    if not reference.strip():
        return 0.0
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def token_f1_score(prediction: str, reference: str) -> float:
    """SQuAD-style F1 over normalized word tokens."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (dynamic program)."""
    # Keep only the previous row: O(len(b)) memory.
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l_score(prediction: str, reference: str) -> float:
    """ROUGE-L F1 over normalized tokens (beta = 1)."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def alias_coverage_score(answer: str, aliases: list[str]) -> float | None:
    """Fraction of acceptable answer strings found in the answer.

    Returns None when the item lists no aliases, so the aggregate
    can skip it instead of scoring a vacuous 0.
    """
    if not aliases:
        return None
    normalized_answer = normalize_text(answer)
    hits = sum(
        1 for alias in aliases
        if normalize_text(alias) in normalized_answer
    )
    return hits / len(aliases)


def citation_scores(
    predicted: list[str], expected: list[str]
) -> tuple[float | None, float | None, float | None]:
    """Precision/recall/F1 of predicted source labels vs expected.

    Labels compare case-insensitively after stripping whitespace.
    Returns (None, None, None) when the item lists no expected
    sources — there is nothing to cite-check against.
    """
    if not expected:
        return None, None, None
    expected_norm = {s.strip().lower() for s in expected if s.strip()}
    predicted_norm = {s.strip().lower() for s in predicted if s.strip()}
    if not expected_norm:
        return None, None, None
    hits = len(expected_norm & predicted_norm)
    precision = hits / len(predicted_norm) if predicted_norm else 0.0
    recall = hits / len(expected_norm)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1


def parse_bracket_labels(answer: str) -> list[str]:
    """Source labels written as ``[PMID:38291023]`` in answer text.

    Plain ``[1]``-style numeric citations carry no source identity,
    so they are skipped — they are checked by ``prompts.py``'s
    ``validate_citations`` instead.
    """
    labels = []
    for match in _BRACKET_LABEL_RE.finditer(answer):
        label = match.group(1).strip()
        if label and not label.isdigit():
            labels.append(label)
    return labels


def is_abstention(answer: str) -> bool:
    """True when the answer declines to answer instead of guessing."""
    normalized = normalize_text(answer)
    return any(
        normalize_text(phrase) in normalized for phrase in ABSTENTION_PHRASES
    )


def score_item(
    item: EvalItem,
    prediction: Prediction | None,
    correct_f1: float = DEFAULT_CORRECT_F1,
) -> ItemResult:
    """Score one prediction against one eval item.

    A missing prediction scores 0 on every metric (and counts as a
    failure to abstain on unanswerable items) — silently dropping it
    would flatter the model.
    """
    answer = prediction.answer if prediction else ""
    citations = prediction.citations if prediction else []
    abstained = is_abstention(answer)

    if item.answerable:
        em = exact_match_score(answer, item.reference)
        f1 = token_f1_score(answer, item.reference)
        rouge = rouge_l_score(answer, item.reference)
        coverage = alias_coverage_score(answer, item.aliases)
        correct = bool(
            em == 1.0
            or (coverage is not None and coverage > 0.0)
            or f1 >= correct_f1
        )
    else:
        # Unanswerable: there is no reference to match, so only
        # abstention counts. Any substantive answer is wrong.
        em = f1 = rouge = 0.0
        coverage = None
        correct = abstained

    precision, recall, cit_f1 = citation_scores(citations, item.expected_sources)
    return ItemResult(
        item_id=item.item_id,
        question=item.question,
        exact_match=em,
        token_f1=f1,
        rouge_l_f1=rouge,
        alias_coverage=coverage,
        citation_precision=precision,
        citation_recall=recall,
        citation_f1=cit_f1,
        abstained=abstained,
        correct=correct,
    )


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def evaluate(
    items: list[EvalItem],
    predictions: list[Prediction],
    correct_f1: float = DEFAULT_CORRECT_F1,
) -> EvalReport:
    """Score every eval item against predictions matched by question.

    Questions match on normalized text, so a prediction file that
    rephrases whitespace or case still lines up. Duplicate
    predictions for one question keep the first and are counted as
    unmatched; questions with no prediction score 0.
    """
    by_question: dict[str, Prediction] = {}
    n_unmatched = 0
    for pred in predictions:
        key = normalize_text(pred.question)
        if key in by_question:
            n_unmatched += 1
            continue
        by_question[key] = pred

    results: list[ItemResult] = []
    n_missing = 0
    for item in items:
        pred = by_question.pop(normalize_text(item.question), None)
        if pred is None:
            n_missing += 1
        results.append(score_item(item, pred, correct_f1=correct_f1))
    n_unmatched += len(by_question)

    answerable = [r for r, i in zip(results, items) if i.answerable]
    unanswerable = [r for r, i in zip(results, items) if not i.answerable]
    abstention_accuracy = (
        mean([1.0 if r.abstained else 0.0 for r in unanswerable])
        if unanswerable
        else None
    )
    return EvalReport(
        n_items=len(items),
        n_scored=len(items) - n_missing,
        n_missing_predictions=n_missing,
        n_unmatched_predictions=n_unmatched,
        exact_match=mean([r.exact_match for r in results]) if results else 0.0,
        token_f1=mean([r.token_f1 for r in results]) if results else 0.0,
        rouge_l_f1=mean([r.rouge_l_f1 for r in results]) if results else 0.0,
        alias_coverage=_mean_or_none([r.alias_coverage for r in results]),
        citation_precision=_mean_or_none(
            [r.citation_precision for r in results]
        ),
        citation_recall=_mean_or_none([r.citation_recall for r in results]),
        citation_f1=_mean_or_none([r.citation_f1 for r in results]),
        abstention_accuracy=abstention_accuracy,
        accuracy=(
            mean([1.0 if r.correct else 0.0 for r in results])
            if results
            else 0.0
        ),
        items=results,
    )


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, one JSON object per non-empty line."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def load_eval_set(path: Path) -> list[EvalItem]:
    """Load and validate an eval-set JSONL file."""
    return [
        EvalItem.from_record(record, line_no)
        for line_no, record in enumerate(load_jsonl(path), start=1)
    ]


def load_predictions(path: Path) -> list[Prediction]:
    """Load and validate a predictions JSONL file."""
    return [
        Prediction.from_record(record, line_no)
        for line_no, record in enumerate(load_jsonl(path), start=1)
    ]


def demo_predictions(items: list[EvalItem]) -> list[Prediction]:
    """Deterministic stub answers for the built-in demo set.

    Deliberately mixed: the first item is answered correctly with a
    citation, the second is answered with a wrong range, and the
    unanswerable third is met with an abstention. This is a smoke
    test for the scoring code, not a model — the stub never sees a
    real checkpoint.
    """
    stub = {
        normalize_text("Which organelle produces most of a cell's ATP?"): Prediction(
            question="Which organelle produces most of a cell's ATP?",
            answer="The mitochondria produce most of a cell's ATP.",
            citations=["Alberts et al., Molecular Biology of the Cell"],
        ),
        normalize_text(
            "What is the normal fasting blood glucose range "
            "for a non-diabetic adult?"
        ): Prediction(
            question="What is the normal fasting blood glucose range "
            "for a non-diabetic adult?",
            answer="The normal fasting range is 120-140 mg/dL.",
            citations=[],
        ),
        normalize_text(
            "What was the 2026 overall survival of drug X in trial Y?"
        ): Prediction(
            question="What was the 2026 overall survival of drug X in trial Y?",
            answer="제공된 문헌에서는 확인할 수 없습니다.",
            citations=[],
        ),
    }
    return [stub[normalize_text(item.question)] for item in items]


def format_summary(report: EvalReport) -> str:
    """One-page human-readable summary of the aggregate scores."""

    def pct(value: float | None) -> str:
        return f"{value * 100:.1f}%" if value is not None else "n/a"

    lines = [
        f"items: {report.n_items} "
        f"(scored {report.n_scored}, "
        f"missing predictions {report.n_missing_predictions}, "
        f"unmatched predictions {report.n_unmatched_predictions})",
        f"accuracy:          {pct(report.accuracy)}",
        f"exact match:       {pct(report.exact_match)}",
        f"token F1:          {pct(report.token_f1)}",
        f"ROUGE-L F1:        {pct(report.rouge_l_f1)}",
        f"alias coverage:    {pct(report.alias_coverage)}",
        f"citation P/R/F1:   {pct(report.citation_precision)} / "
        f"{pct(report.citation_recall)} / {pct(report.citation_f1)}",
        f"abstention acc.:   {pct(report.abstention_accuracy)}",
    ]
    return "\n".join(lines)


def format_failures(report: EvalReport) -> str:
    """Per-question lines for every item the model got wrong."""
    lines = ["failures:"]
    for result in report.items:
        if result.correct:
            continue
        lines.append(
            f"  - [{result.item_id}] {result.question} "
            f"(EM {result.exact_match:.2f}, F1 {result.token_f1:.2f}, "
            f"abstained={str(result.abstained).lower()})"
        )
    return "\n".join(lines) if len(lines) > 1 else "no failures."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score biomedical QA predictions against a gold set."
    )
    parser.add_argument("--eval-set", type=Path,
                        help="Eval-set JSONL (default: built-in demo set)")
    parser.add_argument("--predictions", type=Path,
                        help="Model predictions JSONL to score")
    parser.add_argument("--demo", action="store_true",
                        help="Score the deterministic stub answers "
                        "on the built-in demo set (smoke test)")
    parser.add_argument("--output", type=Path,
                        help="Write the full JSON report here")
    parser.add_argument("--correct-f1", type=float,
                        default=DEFAULT_CORRECT_F1,
                        help="Token F1 at/above which an answerable "
                        "item counts correct (default 0.5)")
    parser.add_argument("--fail-under", type=float, default=None,
                        help="Exit non-zero when token F1 is below this "
                        "value (CI gate)")
    args = parser.parse_args(argv)

    if args.demo and args.predictions:
        parser.error("--demo and --predictions cannot be combined")
    if not args.demo and args.predictions is None:
        parser.error("give --predictions, or use --demo for a smoke test")

    if args.eval_set is not None:
        items = load_eval_set(args.eval_set)
    else:
        items = [EvalItem.from_record(r, i + 1)
                 for i, r in enumerate(SAMPLE_EVAL_SET)]

    if args.demo:
        predictions = demo_predictions(items)
    else:
        predictions = load_predictions(args.predictions)  # type: ignore[arg-type]

    report = evaluate(items, predictions, correct_f1=args.correct_f1)
    print(format_summary(report))
    print(format_failures(report))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"report written to {args.output}")

    if args.fail_under is not None and report.token_f1 < args.fail_under:
        print(
            f"token F1 {report.token_f1:.3f} is below --fail-under "
            f"{args.fail_under}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
