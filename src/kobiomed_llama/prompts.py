"""Prompt templates for KoBioMed-Llama biomedical QA.

This module owns every string the model sees. It is the counterpart of
``retriever.py``: the retriever fetches the passages, this module turns
a question plus those passages into a prompt the model can answer from
— with citations, in Korean, and without inventing sources.

Design rules the templates enforce:

- **Cite or stay silent.** Every factual claim in the answer must carry
  a ``[n]`` citation pointing at a numbered passage in the context.
  If the context does not support an answer, the model must say so
  instead of guessing.
- **Korean first.** Questions and answers default to Korean; the
  template tells the model to match the question's language.
- **Provenance survives.** Context blocks are headed by the same
  citation labels ``retriever.format_context`` produces
  (``[1] PMID:38291023``), so a ``[1]`` in the answer always maps back
  to a retrievable source.
- **No model required.** Everything here is plain string building, so
  the templates can be unit-tested and reused by the fine-tuning data
  script (``prepare_data.py``) without loading any weights.

Two prompt families live here:

1. **RAG prompts** (``build_rag_prompt``) — used at inference time by
   the CLI: system prompt + retrieved context + question.
2. **Instruction-tuning formats** (``format_instruction_example``,
   ``format_chat_messages``) — used to build training JSONL: an
   Alpaca-style instruction example and a Llama-3 chat rendering of a
   message list, for the LoRA script (``train_lora.py``).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

#: Maximum context characters copied into a prompt. Long retrieved
#: passages are truncated (with a marker) rather than silently dropped,
#: so the model knows the passage was cut.
MAX_CONTEXT_CHARS = 12_000

#: Maximum question characters accepted; longer questions are rejected
#: rather than truncated, because cutting a question changes its meaning.
MAX_QUESTION_CHARS = 2_000

#: Languages the templates explicitly support. Anything else falls back
#: to "answer in the language of the question".
SUPPORTED_LANGUAGES = ("ko", "en")

SYSTEM_PROMPT = """You are KoBioMed-Llama, a Korean biomedical research assistant for life-science researchers at Chungnam National University and Mitos Therapeutics.

Answer in Korean by default; if the question is in another language, answer in that language.

Rules:
- Base your answer ONLY on the passages in <context>. Every factual claim must cite the passage it comes from as [n], using the passage numbers shown in the context.
- If the context does not contain enough information to answer, say so explicitly ("제공된 문헌에서는 확인할 수 없습니다") instead of guessing. Do not invent citations, PMIDs, or study results.
- Distinguish established findings from hypotheses: say what the literature shows, and say separately what remains uncertain or untested.
- Use precise biomedical terminology (gene/protein names, pathway names) and keep the answer concise. Prefer a short structured answer (key point first, then supporting detail) over a long essay.
- Do not repeat the context verbatim; synthesize it into an answer to the question asked."""

#: System prompt used when no retrieved context is available. The model
#: answers from its own knowledge but must flag uncertainty and must not
#: fabricate citations — a weaker guarantee than the RAG prompt, stated
#: up front so callers can prefer the RAG path whenever possible.
NO_CONTEXT_SYSTEM_PROMPT = """You are KoBioMed-Llama, a Korean biomedical research assistant for life-science researchers.

No literature passages were retrieved for this question, so answer from your general biomedical knowledge.

Rules:
- Answer in Korean by default; if the question is in another language, answer in that language.
- Say explicitly that no supporting literature was retrieved, and mark any claim you are unsure about as uncertain.
- Do NOT invent citations, PMIDs, DOIs, or study results. If you cannot recall a source, say so.
- Distinguish established findings from hypotheses, and keep the answer concise."""

RAG_QA_TEMPLATE = """<context>
{context}
</context>

<question>
{question}
</question>

Answer the question using only the passages above. Cite each factual claim with the passage number, e.g. [1], [2]. If the passages do not contain the answer, say "제공된 문헌에서는 확인할 수 없습니다" and briefly explain what is missing."""

SUMMARY_TEMPLATE = """<context>
{context}
</context>

Summarize the passages above in Korean, in 3-5 bullet points. Each bullet must cite the passage(s) it summarizes, e.g. [1]. Preserve key quantitative results (effect sizes, p-values, sample sizes) exactly as stated; do not round or reinterpret them."""

MECHANISM_TEMPLATE = """<context>
{context}
</context>

<question>
{question}
</question>

Explain the molecular mechanism behind the question above, step by step, using only the passages in the context. Name the molecules, pathways, and cell types involved, cite each step with its passage number, e.g. [1], and end with one sentence on what remains unknown or untested."""

#: Instruction-tuning system prompt: shorter than the RAG system prompt
#: because the training examples already carry the context.
INSTRUCTION_SYSTEM_PROMPT = (
    "You are KoBioMed-Llama, a Korean biomedical research assistant. "
    "Answer in Korean, cite retrieved passages as [n], and say "
    '"제공된 문헌에서는 확인할 수 없습니다" when the context does not '
    "contain the answer."
)

TemplateName = Literal["qa", "summary", "mechanism"]

TEMPLATES: dict[TemplateName, str] = {
    "qa": RAG_QA_TEMPLATE,
    "summary": SUMMARY_TEMPLATE,
    "mechanism": MECHANISM_TEMPLATE,
}

#: Llama-3 chat-format markers used by ``format_chat_messages``.
LLAMA3_BOS = "<|begin_of_text|>"
LLAMA3_HEADER = "<|start_header_id|>{role}<|end_header_id|>\n\n"
LLAMA3_EOT = "<|eot_id|>"


@dataclass(frozen=True)
class ChatMessage:
    """One turn of a chat conversation (``role`` + ``content``)."""

    role: Literal["system", "user", "assistant"]
    content: str


def _require_non_empty(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def truncate_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Truncate over-long context, marking the cut explicitly.

    A truncated passage keeps its beginning (where the key claim usually
    sits) and ends with ``[... truncated]`` so the model does not mistake
    a cut-off sentence for a complete statement.
    """
    if len(context) <= max_chars:
        return context
    return context[:max_chars].rstrip() + "\n[... truncated]"


def build_rag_prompt(
    question: str,
    context: str,
    template: TemplateName = "qa",
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Build a full RAG prompt: system prompt + templated user turn.

    Args:
        question: The researcher's question (max ``MAX_QUESTION_CHARS``).
        context: Numbered passages, e.g. the output of
            ``retriever.format_context`` — ``"[1] PMID:38291023\\n..."``.
        template: Which task template to fill (``"qa"``, ``"summary"``,
            ``"mechanism"``).
        system_prompt: Override for the default system prompt.

    Returns:
        ``"<system>\\n\\n<user turn>"`` — ready to send to a chat model.
    """
    _require_non_empty(question, "question")
    _require_non_empty(context, "context")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"question is {len(question)} chars, over the "
            f"{MAX_QUESTION_CHARS}-char limit"
        )
    user_turn = TEMPLATES[template].format(
        context=truncate_context(context), question=question.strip()
    )
    return f"{system_prompt.strip()}\n\n{user_turn}"


def build_rag_prompt_from_hits(
    question: str,
    hits: Sequence[object],
    template: TemplateName = "qa",
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Build a RAG prompt directly from retrieved hits.

    ``hits`` are duck-typed: anything with ``.citation`` and ``.text``
    attributes works, so ``retriever.RetrievedChunk`` objects can be
    passed without importing the retriever. An empty hit list falls back
    to the no-context system prompt with an explicit "no literature"
    note, so the call never silently produces a context-free QA prompt.
    """
    if not hits:
        return (
            f"{NO_CONTEXT_SYSTEM_PROMPT.strip()}\n\n"
            f"<question>\n{question.strip()}\n</question>\n\n"
            "No literature passages were retrieved for this question."
        )
    blocks = []
    for i, hit in enumerate(hits, start=1):
        citation = getattr(hit, "citation", "unknown")
        text = getattr(hit, "text", "")
        blocks.append(f"[{i}] {citation}\n{text}")
    return build_rag_prompt(
        question, "\n\n".join(blocks), template=template,
        system_prompt=system_prompt,
    )


def build_no_context_prompt(question: str) -> str:
    """Build a prompt for a question with no retrieved context.

    Used when the retriever found nothing (empty store or everything
    below ``min_score``): the model answers from general knowledge with
    the no-context rules, rather than receiving a QA prompt whose
    context block would be empty.
    """
    _require_non_empty(question, "question")
    return (
        f"{NO_CONTEXT_SYSTEM_PROMPT.strip()}\n\n"
        f"<question>\n{question.strip()}\n</question>"
    )


def format_instruction_example(
    instruction: str,
    output: str,
    input_text: str = "",
    system_prompt: str = INSTRUCTION_SYSTEM_PROMPT,
) -> dict[str, str]:
    """Format one Alpaca-style instruction-tuning example.

    The returned dict (``system`` / ``instruction`` / ``input`` /
    ``output``) is the record shape ``prepare_data.py`` writes to JSONL.
    ``input_text`` holds optional retrieved context; empty for
    knowledge-only examples.
    """
    _require_non_empty(instruction, "instruction")
    _require_non_empty(output, "output")
    return {
        "system": system_prompt,
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "output": output.strip(),
    }


def format_chat_messages(
    messages: Sequence[ChatMessage],
    add_generation_prompt: bool = True,
) -> str:
    """Render messages in the Llama-3 chat format.

    Produces the ``<|begin_of_text|>`` / ``<|start_header_id|>`` /
    ``<|eot_id|>`` rendering that Llama-3-family models (including the
    KoBioMed-Llama base) were trained on. With ``add_generation_prompt``
    the output ends with an open ``assistant`` header, ready for
    generation; without it, it is a complete transcript.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    parts = [LLAMA3_BOS]
    for message in messages:
        _require_non_empty(message.content, f"{message.role} content")
        parts.append(LLAMA3_HEADER.format(role=message.role))
        parts.append(message.content.strip())
        parts.append(LLAMA3_EOT)
    if add_generation_prompt:
        parts.append(LLAMA3_HEADER.format(role="assistant"))
    return "".join(parts)


_CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_citations(answer: str) -> list[int]:
    """Extract the passage numbers a generated answer cites.

    Returns the cited ``[n]`` numbers in order of appearance, deduplicated.
    Used by the evaluation script to check that answers actually cite
    their sources — and that every cited number exists in the context.
    """
    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def validate_citations(answer: str, n_context_passages: int) -> list[str]:
    """Check an answer's citations against the context it was given.

    Returns a list of problems (empty = clean): citations pointing at
    passages that do not exist (``[9]`` with only 3 passages), and the
    absence of any citation at all. A model that answers without citing
    is not following the template, so that counts as a problem too.
    """
    problems: list[str] = []
    cited = parse_citations(answer)
    if not cited:
        problems.append("answer contains no citations")
    for number in cited:
        if number < 1 or number > n_context_passages:
            problems.append(
                f"citation [{number}] has no matching passage "
                f"(context has {n_context_passages})"
            )
    return problems


def list_templates() -> list[str]:
    """Names of the available RAG task templates."""
    return sorted(TEMPLATES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a KoBioMed-Llama prompt from a question "
        "and retrieved context."
    )
    parser.add_argument("--question", default=None,
                        help="Question to ask (default: read from stdin)")
    parser.add_argument("--context-file", default=None,
                        help="File with numbered context passages "
                        "(e.g. retriever.py output)")
    parser.add_argument("--template", choices=list_templates(), default="qa")
    parser.add_argument("--no-context", action="store_true",
                        help="Build the no-context prompt instead")
    parser.add_argument("--list-templates", action="store_true",
                        help="Print available template names and exit")
    parser.add_argument("--chat", action="store_true",
                        help="Render the prompt in Llama-3 chat format")
    args = parser.parse_args()

    if args.list_templates:
        print("\n".join(list_templates()))
        return

    question = args.question
    if question is None:
        question = sys.stdin.read()
    if not question.strip():
        raise ValueError("no question given (--question or stdin)")

    if args.no_context:
        prompt = build_no_context_prompt(question)
    else:
        if args.context_file is None:
            raise ValueError(
                "--context-file is required unless --no-context is given"
            )
        context = Path(args.context_file).read_text(encoding="utf-8")
        prompt = build_rag_prompt(question, context, template=args.template)

    if args.chat:
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt.split("\n\n", 1)[1]),
        ]
        prompt = format_chat_messages(messages)
    print(prompt)


if __name__ == "__main__":
    main()
