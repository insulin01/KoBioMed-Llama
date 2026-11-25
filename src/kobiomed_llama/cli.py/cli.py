"""Command-line entry point for KoBioMed-Llama RAG question answering.

This is the piece a researcher actually runs: give it a question, it
retrieves the relevant literature passages from the FAISS vector store
(``vectorstore.py``), assembles a cited RAG prompt (``prompts.py``),
generates an answer with a causal language model, and prints the
answer together with its sources.

Pipeline (one question, end to end)::

    question -> Retriever.retrieve -> build_chat_messages
             -> Generator.generate -> AnswerResult -> print

Design rules, inherited from the rest of the pipeline:

- **Cite or stay silent.** When the store returns no passages, the CLI
  abstains with the canonical "제공된 문헌에서는 확인할 수 없습니다"
  instead of answering from the model's general knowledge. Pass
  ``--allow-no-context`` to use the no-context prompt instead; the
  answer is then explicitly flagged as unsupported by retrieved
  literature.
- **The prompt is inspectable.** ``--show-prompt`` prints exactly what
  the model would receive (Llama-3 chat rendering) and exits without
  loading any model weights — useful for debugging retrieval and for
  machines without a GPU.
- **Batch mode feeds the evaluator.** ``--batch`` reads a questions
  JSONL file and writes predictions JSONL in the shape
  ``evals/evaluate_qa.py`` scores (``question`` / ``answer`` /
  ``citations`` with source labels), so one command produces the
  input for the next stage of the pipeline.
- **Heavy dependencies stay lazy.** ``transformers`` / ``torch`` /
  ``peft`` are imported only when a model is actually loaded, so
  prompt inspection and argument errors fail fast without them.

The default base model is the gated Llama-3.1-8B-Instruct checkpoint
this project fine-tunes (see ``train_lora.py``); pass ``--model`` with
a local path or a public checkpoint when that is unavailable, and
``--adapter`` with a LoRA adapter directory from ``train_lora.py`` to
answer with the fine-tuned model.

Example usage::

    python cli.py data/vectorstore "PHF20은 미토콘드리아 생합성에 어떤 역할을 하나요?"
    python cli.py data/vectorstore --show-prompt "미토콘드리아 ATP 생성 장소는?"
    python cli.py data/vectorstore --batch evals/sample_qa.jsonl --output runs/predictions.jsonl --model Qwen/Qwen2.5-0.5B-Instruct
    python cli.py data/vectorstore --adapter runs/lora-smoke
    python cli.py data/vectorstore

With no question argument the CLI reads questions interactively from
stdin, one per line; an empty line ends the session.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from embeddings import DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL
from embeddings import Embedder
from prompts import (
    NO_CONTEXT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TEMPLATES,
    ChatMessage,
    TemplateName,
    format_chat_messages,
    list_templates,
    parse_citations,
    truncate_context,
    validate_citations,
)
from retriever import DEFAULT_K, Retriever, RetrievedChunk
from vectorstore import VectorStore

#: Default base model: the Llama-3.1 instruct checkpoint this project
#: fine-tunes. Gated on Hugging Face — pass --model for a public or
#: local checkpoint when access is unavailable.
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

#: Default generation length. RAG answers are short by design (the
#: system prompt asks for a concise structured answer).
DEFAULT_MAX_NEW_TOKENS = 512

#: Answer used when no passages were retrieved and --allow-no-context
#: was not given. Starts with the canonical abstention phrase
#: ``prompts.py`` prescribes, so ``evaluate_qa.py`` recognises it.
ABSTENTION_ANSWER = (
    "제공된 문헌에서는 확인할 수 없습니다. "
    "벡터 스토어에서 관련 문헌을 찾지 못했습니다."
)


@dataclass
class SourceInfo:
    """One retrieved passage behind an answer, with its provenance."""

    citation: str
    score: float
    text: str


@dataclass
class AnswerResult:
    """Everything one answered question produced.

    ``answer`` is empty for a prompt-only run (``--show-prompt``);
    ``abstained`` marks the no-context refusal path; ``prompt`` holds
    the exact chat-rendered prompt the model received (or would have
    received), for debugging and reproducibility.
    """

    question: str
    answer: str
    citations: list[int] = field(default_factory=list)
    sources: list[SourceInfo] = field(default_factory=list)
    citation_problems: list[str] = field(default_factory=list)
    prompt: str = ""
    abstained: bool = False
    model: str = ""

    @property
    def n_context_passages(self) -> int:
        return len(self.sources)

    def to_dict(self, include_prompt: bool = False) -> dict[str, Any]:
        """Plain dict for JSON output (``--json`` / batch predictions)."""
        data = asdict(self)
        if not include_prompt:
            data.pop("prompt", None)
        return data

    def to_prediction(self) -> dict[str, Any]:
        """Prediction record in the shape ``evaluate_qa.py`` scores.

        ``citations`` carries source labels (e.g. ``PMID:38291023``)
        rather than passage numbers, because the evaluator matches
        them against each item's ``expected_sources``.
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [source.citation for source in self.sources],
        }


def load_tokenizer(model_name: str) -> Any:
    """Load the tokenizer; fall back to EOS as the pad token.

    Same rule as ``train_lora.load_tokenizer``: Llama-3 and GPT-2-style
    tokenizers ship without a pad token, and generation needs one.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _stop_token_ids(tokenizer: Any) -> list[int]:
    """Token ids that end generation: EOS plus Llama-3's turn end.

    ``<|eot_id|>`` is only added when the tokenizer actually defines
    it as an added token — for a GPT-2 tokenizer the string is not a
    token at all, and converting it would silently yield a wrong id.
    """
    stop_ids = [tokenizer.eos_token_id]
    added = tokenizer.get_added_vocab()
    if "<|eot_id|>" in added:
        stop_ids.append(added["<|eot_id|>"])
    return [token_id for token_id in stop_ids if token_id is not None]


class Generator:
    """A causal language model that answers from a chat prompt.

    The model and tokenizer load lazily on first use, so constructing
    a ``Generator`` is cheap and argument validation can fail before
    any weights are downloaded. Pass ``adapter`` with a LoRA adapter
    directory (as written by ``train_lora.py``) to answer with the
    fine-tuned model instead of the base checkpoint.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BASE_MODEL,
        adapter: str | None = None,
        device: str = "auto",
        bf16: bool = False,
    ) -> None:
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"device must be 'auto', 'cpu' or 'cuda', got {device!r}"
            )
        self.model_name = model_name
        self.adapter = adapter
        self.device = device
        self.bf16 = bf16
        self._tokenizer: Any = None
        self._model: Any = None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.model_name)
        return self._tokenizer

    @property
    def model(self) -> Any:
        """Load the base model (and adapter) on first access."""
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM

            use_cuda = torch.cuda.is_available()
            if self.device == "cuda" and not use_cuda:
                raise RuntimeError(
                    "device 'cuda' was requested but no CUDA device "
                    "is available"
                )
            dtype = torch.bfloat16 if self.bf16 else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map=(
                    "auto"
                    if (self.device == "auto" and use_cuda)
                    or self.device == "cuda"
                    else None
                ),
            )
            model.eval()
            if self.adapter is not None:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter)
                model.eval()
            self._model = model
        return self._model

    def generate(
        self,
        messages: Sequence[ChatMessage],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Generate a completion for chat ``messages``.

        ``temperature=0`` (the default) is greedy decoding — the
        reproducible choice for a QA tool. Sampling needs
        ``temperature > 0``; ``top_p`` is ignored for greedy decoding.
        Returns the decoded new tokens only (the prompt is not echoed).
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0.0:
            raise ValueError("temperature must not be negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must satisfy 0 < top_p <= 1")

        import torch

        rendered = format_chat_messages(list(messages))
        inputs = self.tokenizer(rendered, return_tensors="pt").to(
            self.model.device
        )
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": _stop_token_ids(self.tokenizer),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            generate_kwargs.update(
                {"do_sample": True, "temperature": temperature,
                 "top_p": top_p}
            )
        else:
            generate_kwargs["do_sample"] = False
        with torch.no_grad():
            output = self.model.generate(**inputs, **generate_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()


def build_chat_messages(
    question: str,
    hits: Sequence[RetrievedChunk],
    template: TemplateName = "qa",
) -> list[ChatMessage]:
    """Build the system + user messages for one question.

    Mirrors ``prompts.build_rag_prompt_from_hits`` but returns chat
    messages instead of a flat string, so the CLI can render them in
    Llama-3 chat format for generation. An empty hit list yields the
    no-context system prompt; the caller decides whether to generate
    from it or abstain.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if not hits:
        return [
            ChatMessage(role="system", content=NO_CONTEXT_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=(
                    "<question>\n" + question.strip() + "\n</question>"
                ),
            ),
        ]
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] {hit.citation}\n{hit.text}")
    context = truncate_context("\n\n".join(blocks))
    user_turn = TEMPLATES[template].format(
        context=context, question=question.strip()
    )
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_turn),
    ]


def answer_question(
    question: str,
    retriever: Retriever,
    generator: Generator | None = None,
    template: TemplateName = "qa",
    k: int | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
    allow_no_context: bool = False,
    show_prompt: bool = False,
) -> AnswerResult:
    """Answer one question: retrieve, prompt, generate, validate.

    With ``generator=None`` or ``show_prompt=True`` no generation
    happens and the result carries the prompt only — the retrieval
    and prompt-assembly half of the pipeline, testable without model
    weights. With no retrieved passages and ``allow_no_context=False``
    the result is the abstention answer, without calling the model.
    """
    hits = retriever.retrieve(question, k=k)
    messages = build_chat_messages(question, hits, template=template)
    prompt = format_chat_messages(messages)
    sources = [
        SourceInfo(citation=hit.citation, score=hit.score, text=hit.text)
        for hit in hits
    ]
    model_name = generator.model_name if generator is not None else ""

    if not hits and not allow_no_context:
        return AnswerResult(
            question=question.strip(),
            answer=ABSTENTION_ANSWER,
            sources=[],
            citation_problems=[],
            prompt=prompt,
            abstained=True,
            model=model_name,
        )
    if generator is None or show_prompt:
        return AnswerResult(
            question=question.strip(),
            answer="",
            sources=sources,
            citation_problems=[],
            prompt=prompt,
            abstained=False,
            model=model_name,
        )

    answer = generator.generate(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    problems = validate_citations(answer, len(hits))
    return AnswerResult(
        question=question.strip(),
        answer=answer,
        citations=parse_citations(answer),
        sources=sources,
        citation_problems=problems,
        prompt=prompt,
        abstained=False,
        model=model_name,
    )


def format_answer(
    result: AnswerResult,
    show_sources: bool = True,
    show_prompt: bool = False,
) -> str:
    """Render an answer for the terminal: answer, then its sources.

    Citation problems (an answer citing a passage that does not exist,
    or citing nothing at all) are printed as a warning, not hidden —
    a RAG answer that does not cite its context is not following the
    template, and the reader should know.
    """
    lines = [result.answer]
    if result.abstained:
        lines.append("(관련 문헌 없음 — 모델에 질문하지 않았습니다)")
    if result.citation_problems:
        lines.append("")
        lines.append("[인용 경고]")
        for problem in result.citation_problems:
            lines.append(f"  - {problem}")
    if show_sources and result.sources:
        lines.append("")
        lines.append("출처:")
        for i, source in enumerate(result.sources, start=1):
            lines.append(
                f"  [{i}] {source.citation} (score {source.score:.3f})"
            )
    if show_prompt:
        lines.append("")
        lines.append("--- prompt ---")
        lines.append(result.prompt)
    return "\n".join(lines)


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Read a batch questions file: one JSON object per line.

    Each record needs ``question``; ``id`` is kept when present so
    predictions can be matched back to the eval set.
    """
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_no}: expected a JSON object"
                )
            question = str(record.get("question", "")).strip()
            if not question:
                raise ValueError(f"{path}:{line_no}: missing 'question'")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no questions found")
    return records


def run_batch(
    questions: list[dict[str, Any]],
    retriever: Retriever,
    generator: Generator,
    template: TemplateName,
    output_path: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    allow_no_context: bool,
) -> list[AnswerResult]:
    """Answer every question and write predictions JSONL.

    Writes one prediction record per question (same order as the
    input), in the shape ``evaluate_qa.py`` scores. Progress goes to
    stderr so stdout stays clean for piping. Returns the results for
    the caller's summary.
    """
    results: list[AnswerResult] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(questions, start=1):
            result = answer_question(
                str(record["question"]),
                retriever,
                generator=generator,
                template=template,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                allow_no_context=allow_no_context,
            )
            prediction = result.to_prediction()
            if record.get("id") is not None:
                prediction = {"id": record["id"], **prediction}
            handle.write(
                json.dumps(prediction, ensure_ascii=False) + "\n"
            )
            results.append(result)
            print(
                f"[{index}/{len(questions)}] "
                f"{'abstained' if result.abstained else 'answered'}: "
                f"{result.question[:60]}",
                file=sys.stderr,
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask KoBioMed-Llama a biomedical question "
        "with RAG over a FAISS vector store."
    )
    parser.add_argument(
        "store", help="Vector store stem (from vectorstore.py)")
    parser.add_argument(
        "question", nargs="?", default=None,
        help="Question to ask (omit for interactive mode)")
    parser.add_argument(
        "--model", default=DEFAULT_BASE_MODEL,
        help="Causal LM checkpoint for generation "
        f"(default: {DEFAULT_BASE_MODEL})")
    parser.add_argument(
        "--adapter", default=None,
        help="LoRA adapter directory from train_lora.py "
        "(default: use the base model as-is)")
    parser.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model for the query "
        f"(default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Device for generation (default: auto)")
    parser.add_argument(
        "--bf16", action="store_true",
        help="Load the model in bfloat16 (GPU only; default: float32)")
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Passages retrieved per question (default: {DEFAULT_K})")
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="Drop retrieved passages below this similarity score")
    parser.add_argument(
        "--template", choices=list_templates(), default="qa",
        help="RAG prompt template (default: qa)")
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum generated tokens (default: {DEFAULT_MAX_NEW_TOKENS})")
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature; 0 means greedy decoding "
        "(default: 0.0)")
    parser.add_argument(
        "--top-p", type=float, default=1.0,
        help="Nucleus sampling cutoff, used when temperature > 0 "
        "(default: 1.0)")
    parser.add_argument(
        "--show-prompt", action="store_true",
        help="Print the assembled prompt and exit without generating")
    parser.add_argument(
        "--allow-no-context", action="store_true",
        help="When nothing is retrieved, answer from the model's "
        "general knowledge instead of abstaining")
    parser.add_argument(
        "--batch", type=Path, default=None,
        help="Questions JSONL file: answer every question and "
        "write predictions JSONL (requires --output)")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Predictions JSONL for --batch")
    parser.add_argument(
        "--json", action="store_true",
        help="Print the single-question result as JSON")
    parser.add_argument(
        "--hide-sources", action="store_true",
        help="Do not print the retrieved source list")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # parse_intermixed_args (not parse_args): the optional positional
    # "question" must be accepted after options too ("cli.py STORE
    # --show-prompt 'question'"). Plain parse_args rejects that order
    # for a nargs="?" positional.
    args = parser.parse_intermixed_args(argv)

    if args.batch is not None and args.question is not None:
        parser.error("--batch cannot be combined with a question")
    if args.batch is not None and args.output is None:
        parser.error("--batch requires --output")
    if args.batch is not None and args.show_prompt:
        parser.error("--batch cannot be combined with --show-prompt")
    if args.temperature < 0.0:
        parser.error("--temperature must not be negative")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must satisfy 0 < top_p <= 1")

    store = VectorStore.load(args.store)
    print(f"loaded vector store: {len(store)} chunk(s)", file=sys.stderr)
    embedder = Embedder(model_name=args.embedding_model)
    retriever = Retriever(
        store, embedder, k=args.k, min_score=args.min_score
    )
    generator = Generator(
        model_name=args.model,
        adapter=args.adapter,
        device=args.device,
        bf16=args.bf16,
    )

    if args.batch is not None:
        questions = load_questions(args.batch)
        results = run_batch(
            questions,
            retriever,
            generator,
            template=args.template,
            output_path=args.output,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            allow_no_context=args.allow_no_context,
        )
        n_abstained = sum(1 for r in results if r.abstained)
        print(
            f"answered {len(results)} question(s), "
            f"{n_abstained} abstention(s) -> {args.output}",
            file=sys.stderr,
        )
        return 0

    if args.question is not None:
        result = answer_question(
            args.question,
            retriever,
            generator=None if args.show_prompt else generator,
            template=args.template,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            allow_no_context=args.allow_no_context,
            show_prompt=args.show_prompt,
        )
        if args.show_prompt:
            print(result.prompt)
        elif args.json:
            print(json.dumps(
                result.to_dict(include_prompt=False),
                ensure_ascii=False, indent=2,
            ))
        else:
            print(format_answer(
                result, show_sources=not args.hide_sources
            ))
        for problem in result.citation_problems:
            print(f"[인용 경고] {problem}", file=sys.stderr)
        return 0

    # Interactive mode: one question per line, empty line quits.
    print(
        "KoBioMed-Llama RAG CLI — 질문을 입력하세요 "
        "(빈 줄: 종료)",
        file=sys.stderr,
    )
    try:
        while True:
            line = input("> ")
            if not line.strip():
                break
            result = answer_question(
                line,
                retriever,
                generator=None if args.show_prompt else generator,
                template=args.template,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                allow_no_context=args.allow_no_context,
                show_prompt=args.show_prompt,
            )
            if args.show_prompt:
                print(result.prompt)
            else:
                print(format_answer(
                    result, show_sources=not args.hide_sources
                ))
            for problem in result.citation_problems:
                print(f"[인용 경고] {problem}", file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
