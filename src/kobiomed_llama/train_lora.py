"""LoRA fine-tuning for KoBioMed-Llama.

Trains a LoRA adapter on top of a Llama-3-family base model using the
instruction JSONL that ``prepare_data.py`` produces (Alpaca-style
``system`` / ``instruction`` / ``input`` / ``output`` records).

Training setup, in one paragraph: each example is rendered with the
Llama-3 chat format from ``prompts.format_chat_messages`` and the loss
is masked to the assistant's reply only (completion-only loss), so the
model learns *how to answer* without being trained to reproduce the
prompt. The base model weights stay frozen; only the LoRA adapter
(``peft``) is updated. The run saves the adapter (and tokenizer) to
``--output-dir``; ``--merge`` additionally writes a merged full model
next to it.

The default base model is a gated Llama-3.1 instruct checkpoint, so a
Hugging Face token with access to it is required (``HF_TOKEN`` env var
or ``huggingface-cli login``). Any causal-LM checkpoint works, though:
pass a small model such as ``hf-internal-testing/tiny-random-gpt2``
for a wiring smoke test (see the ``--dry-run`` / smoke-test notes
below). Nothing here downloads or trains on real data unless you point
it at a real ``--train-file``.

Example usage::

    python train_lora.py --train-file data/train.jsonl --val-file data/val.jsonl --output-dir runs/lora-v1
    python train_lora.py --train-file data/train.jsonl --output-dir runs/lora-v1 --num-train-epochs 2 --lora-r 32
    python train_lora.py --train-file data/train.jsonl --output-dir runs/lora-smoke --base-model hf-internal-testing/tiny-random-gpt2 --target-modules c_attn c_proj --max-steps 2 --max-seq-len 128

Dry run (no model weights, no training — validates the data, the
tokenizer and the LoRA config, and prints token-length stats)::

    python train_lora.py --train-file data/train.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:
    from prompts import ChatMessage, format_chat_messages
except ImportError:  # when imported as part of the kobiomed_llama package
    from .prompts import ChatMessage, format_chat_messages  # type: ignore[no-redef]

#: Default base model: the Llama-3.1 instruct checkpoint this project
#: adapts. Gated — needs a Hugging Face token with access granted.
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

#: Default LoRA rank. r=16 is a common starting point for 8B-class
#: instruction tuning on a few thousand examples; raise it (32/64) if
#: the adapter underfits, lower it (8) for very small datasets.
DEFAULT_LORA_R = 16

#: Default LoRA scaling factor.
DEFAULT_LORA_ALPHA = 32

#: Default LoRA dropout.
DEFAULT_LORA_DROPOUT = 0.05

#: Default target modules for Llama-family models (all linear layers
#: that carry the attention / MLP computation). Override with
#: ``--target-modules`` for other architectures (e.g. ``c_attn c_proj``
#: for GPT-2).
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Default maximum sequence length (tokens). Examples from
#: ``prepare_data.py`` are capped at 6,000 input + 4,000 output chars,
#: so 2,048 tokens covers nearly all of them; longer ones are counted
#: and truncated (see ``tokenize_example``).
DEFAULT_MAX_SEQ_LEN = 2_048

#: Default learning rate for LoRA on an 8B model.
DEFAULT_LEARNING_RATE = 2e-4

#: Default number of training epochs.
DEFAULT_NUM_TRAIN_EPOCHS = 3

#: Default per-device batch size. Kept small on purpose: an 8B model
#: with gradient checkpointing fits on a single 24 GB GPU at batch 2-4
#: with ``--gradient-accumulation-steps`` raised to taste.
DEFAULT_PER_DEVICE_BATCH_SIZE = 2

#: Default gradient accumulation steps (effective batch = 2 * 8 = 16).
DEFAULT_GRAD_ACCUM_STEPS = 8

#: Fixed seed so a re-run over the same data reproduces the same
#: shuffle and the same adapter initialization.
DEFAULT_SEED = 42

#: Fraction of optimizer steps spent warming the learning rate up.
#: transformers 5.x dropped the ``warmup_ratio`` argument, so when it
#: is not supported the same fraction is converted to ``warmup_steps``
#: from the planned step count (see ``train``).
WARMUP_RATIO = 0.03

#: Label value that tells the loss to ignore a token.
IGNORE_INDEX = -100


@dataclass
class TrainingStats:
    """Counts collected while building and running the training job."""

    examples_read: int = 0
    examples_kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    truncated: int = 0
    train_tokens: int = 0
    completion_tokens: int = 0
    trainable_params: int = 0
    total_params: int = 0
    final_train_loss: float | None = None
    eval_loss: float | None = None

    def report(self) -> str:
        lines = [
            f"examples read: {self.examples_read}",
            f"examples kept: {self.examples_kept}",
            f"truncated to max_seq_len: {self.truncated}",
            f"train tokens: {self.train_tokens}",
            f"completion tokens (loss-bearing): {self.completion_tokens}",
        ]
        if self.dropped:
            lines.append("dropped:")
            for reason, count in sorted(self.dropped.items()):
                lines.append(f"  {reason}: {count}")
        if self.total_params:
            pct = 100.0 * self.trainable_params / self.total_params
            lines.append(
                f"trainable params: {self.trainable_params:,} / "
                f"{self.total_params:,} ({pct:.2f}%)"
            )
        if self.final_train_loss is not None:
            lines.append(f"final train loss: {self.final_train_loss:.4f}")
        if self.eval_loss is not None:
            lines.append(
                f"eval loss: {self.eval_loss:.4f} "
                f"(perplexity {math.exp(self.eval_loss):.2f})"
            )
        return "\n".join(lines)


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts (blank lines skipped)."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def example_to_messages(example: dict) -> tuple[list[ChatMessage], str]:
    """Split one Alpaca record into chat messages + assistant reply.

    The user turn carries the instruction plus the optional retrieved
    context (``input``), in the same shape ``prepare_data.py`` writes.
    Returns ``(messages, assistant_text)`` where ``messages`` is the
    system + user turns and ``assistant_text`` is the expected reply.
    """
    system = str(example.get("system", "")).strip()
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "") or "").strip()
    output = str(example.get("output", "")).strip()
    if not instruction:
        raise ValueError("example has an empty instruction")
    if not output:
        raise ValueError("example has an empty output")
    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n<context>\n{input_text}\n</context>"
    messages = [
        ChatMessage(role="system", content=system) if system else None,
        ChatMessage(role="user", content=user_content),
    ]
    return [m for m in messages if m is not None], output


def tokenize_example(
    example: dict,
    tokenizer: Any,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    stats: TrainingStats | None = None,
) -> dict[str, list[int]] | None:
    """Tokenize one example with completion-only loss masking.

    The prompt (system + user turns, ending with the open assistant
    header) and the completion (assistant reply + ``<|eot_id|>``) are
    tokenized *separately* and concatenated, so no tokenizer merge can
    blur the prompt/completion boundary. Labels are ``IGNORE_INDEX``
    over the prompt and the completion token ids over the reply.

    Returns ``None`` (counted in ``stats``) when the prompt alone
    already exceeds ``max_seq_len`` — there would be no room left to
    learn the answer. Otherwise the full sequence is right-truncated
    to ``max_seq_len`` and the truncation is counted.
    """
    messages, assistant_text = example_to_messages(example)
    prompt_text = format_chat_messages(messages, add_generation_prompt=True)
    # The assistant turn rendered on its own, so its tokens are exactly
    # the completion the loss is computed on.
    completion_text = format_chat_messages(
        [*messages, ChatMessage(role="assistant", content=assistant_text)],
        add_generation_prompt=False,
    )[len(prompt_text):]
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False
    )["input_ids"]
    completion_ids = tokenizer(
        completion_text, add_special_tokens=False
    )["input_ids"]
    if len(prompt_ids) >= max_seq_len:
        if stats is not None:
            stats.dropped["prompt alone exceeds max_seq_len"] = (
                stats.dropped.get("prompt alone exceeds max_seq_len", 0) + 1
            )
        return None
    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        if stats is not None:
            stats.truncated += 1
    if stats is not None:
        stats.train_tokens += len(input_ids)
        stats.completion_tokens += sum(
            1 for label in labels if label != IGNORE_INDEX
        )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def build_tokenized_dataset(
    examples: Sequence[dict],
    tokenizer: Any,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    stats: TrainingStats | None = None,
) -> list[dict[str, list[int]]]:
    """Tokenize a list of examples, skipping invalid ones.

    Raises ``ValueError`` if nothing survives — training on an empty
    dataset would silently produce an untrained adapter.
    """
    tokenized: list[dict[str, list[int]]] = []
    for example in examples:
        try:
            item = tokenize_example(
                example, tokenizer, max_seq_len=max_seq_len, stats=stats
            )
        except ValueError as exc:
            if stats is not None:
                stats.dropped[str(exc)] = (
                    stats.dropped.get(str(exc), 0) + 1
                )
            continue
        if item is not None:
            tokenized.append(item)
    if stats is not None:
        stats.examples_kept = len(tokenized)
    if not tokenized:
        raise ValueError(
            "no usable training examples: check --train-file "
            f"(dropped: {dict(stats.dropped) if stats else '?'})"
        )
    return tokenized


class DataCollator:
    """Pad a batch of tokenized examples for causal-LM training.

    Pads ``input_ids`` with the pad token, ``labels`` with
    ``IGNORE_INDEX`` and ``attention_mask`` with 0, then converts to
    tensors. Kept as a small class (rather than
    ``DataCollatorForSeq2SeqLM``) so the padding rule stays explicit
    and the module has no ``datasets`` dependency.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self, features: Sequence[dict[str, list[int]]]
    ) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []
        batch_attention: list[list[int]] = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(
                feature["input_ids"] + [self.pad_token_id] * pad_len
            )
            batch_labels.append(
                feature["labels"] + [IGNORE_INDEX] * pad_len
            )
            batch_attention.append(
                feature["attention_mask"] + [0] * pad_len
            )
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                batch_attention, dtype=torch.long
            ),
        }


def load_tokenizer(model_name: str) -> Any:
    """Load the tokenizer; fall back to EOS as the pad token.

    Llama-3 tokenizers ship without a pad token, and GPT-2-style ones
    too — training needs one for batching, so EOS doubles as pad (the
    collator masks padded positions out of the loss anyway).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_lora_config(
    r: int = DEFAULT_LORA_R,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> Any:
    """Build the ``peft`` LoRA config, validating its values first.

    Validation happens here (not inside the training call) so a bad
    rank or an empty target-module list fails fast in ``--dry-run``
    instead of after the base model has loaded.
    """
    from peft import LoraConfig

    if r <= 0:
        raise ValueError(f"lora_r must be positive, got {r}")
    if alpha <= 0:
        raise ValueError(f"lora_alpha must be positive, got {alpha}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"lora_dropout must satisfy 0 <= d < 1, got {dropout}")
    if not target_modules:
        raise ValueError("target_modules must not be empty")
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def train(args: argparse.Namespace) -> TrainingStats:
    """Run the LoRA training job described by ``args``."""
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
    )

    stats = TrainingStats()
    train_examples = load_jsonl(args.train_file)
    stats.examples_read = len(train_examples)
    val_examples = load_jsonl(args.val_file) if args.val_file else None

    tokenizer = load_tokenizer(args.base_model)
    lora_config = build_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    train_data = build_tokenized_dataset(
        train_examples, tokenizer, max_seq_len=args.max_seq_len, stats=stats
    )
    val_data = (
        build_tokenized_dataset(
            val_examples, tokenizer, max_seq_len=args.max_seq_len
        )
        if val_examples
        else None
    )
    print(stats.report(), file=sys.stderr)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        attn_implementation=(
            "flash_attention_2" if args.flash_attention_2 else "sdpa"
        ),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())
    stats.trainable_params = trainable
    stats.total_params = total

    training_kwargs: dict[str, Any] = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    # transformers 5.x removed warmup_ratio; fall back to the same
    # fraction expressed as warmup_steps over the planned step count.
    if "warmup_ratio" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["warmup_ratio"] = WARMUP_RATIO
    else:
        steps_per_epoch = -(-len(train_data) // (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
        ))
        planned_steps = steps_per_epoch * max(
            1, int(math.ceil(args.num_train_epochs))
        )
        if args.max_steps > 0:
            planned_steps = min(planned_steps, args.max_steps)
        training_kwargs["warmup_steps"] = max(
            1, round(planned_steps * WARMUP_RATIO)
        )
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollator(tokenizer.pad_token_id),
    )
    result = trainer.train()
    stats.final_train_loss = result.training_loss
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved LoRA adapter -> {args.output_dir}", file=sys.stderr)

    if val_data is not None:
        metrics = trainer.evaluate()
        stats.eval_loss = metrics.get("eval_loss")
        print(f"eval: {metrics}", file=sys.stderr)

    if args.merge:
        print("merging adapter into base model...", file=sys.stderr)
        merged = model.merge_and_unload()
        merged_path = str(Path(args.output_dir) / "merged")
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"saved merged model -> {merged_path}", file=sys.stderr)

    print(stats.report(), file=sys.stderr)
    return stats


def dry_run(args: argparse.Namespace) -> TrainingStats:
    """Validate data, tokenizer and LoRA config without training.

    Loads the tokenizer and the training (+ validation) files, builds
    the tokenized dataset, and checks the LoRA config — but never loads
    model weights and never starts an optimizer. Use it to catch bad
    data or bad flags before committing GPU time.
    """
    stats = TrainingStats()
    train_examples = load_jsonl(args.train_file)
    stats.examples_read = len(train_examples)
    val_examples = load_jsonl(args.val_file) if args.val_file else None

    tokenizer = load_tokenizer(args.base_model)
    lora_config = build_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    build_tokenized_dataset(
        train_examples, tokenizer, max_seq_len=args.max_seq_len, stats=stats
    )
    if val_examples:
        build_tokenized_dataset(
            val_examples, tokenizer, max_seq_len=args.max_seq_len
        )
    print(f"tokenizer: {args.base_model} "
          f"(vocab {len(tokenizer)}, pad={tokenizer.pad_token!r})")
    print(f"LoRA config: r={lora_config.r} alpha={lora_config.lora_alpha} "
          f"dropout={lora_config.lora_dropout} "
          f"targets={list(lora_config.target_modules)}")
    print(stats.report())
    print("dry run OK: data, tokenizer and LoRA config are valid; "
          "no model weights loaded, nothing trained.")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune KoBioMed-Llama on instruction JSONL "
        "from prepare_data.py."
    )
    parser.add_argument("--train-file", required=True,
                        help="Train JSONL from prepare_data.py")
    parser.add_argument("--val-file", default=None,
                        help="Validation JSONL (evaluated after training)")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="Base causal-LM checkpoint "
                        f"(default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--output-dir", default="runs/lora-v1",
                        help="Where to save the LoRA adapter")
    parser.add_argument("--max-seq-len", type=int,
                        default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int,
                        default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float,
                        default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--target-modules", nargs="+",
                        default=list(DEFAULT_TARGET_MODULES),
                        help="Linear layers to adapt "
                        "(default: Llama attention + MLP projections)")
    parser.add_argument("--num-train-epochs", type=float,
                        default=DEFAULT_NUM_TRAIN_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Cap optimizer steps (-1 = run full epochs)")
    parser.add_argument("--learning-rate", type=float,
                        default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--per-device-train-batch-size", type=int,
                        default=DEFAULT_PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int,
                        default=DEFAULT_GRAD_ACCUM_STEPS)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bf16", action="store_true",
                        help="Train in bfloat16 (GPU with bf16 support)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Trade compute for memory (recommended for 8B)")
    parser.add_argument("--flash-attention-2", action="store_true",
                        help="Use flash-attention-2 (needs flash-attn)")
    parser.add_argument("--merge", action="store_true",
                        help="Also save the merged full model "
                        "(output_dir/merged)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data, tokenizer and LoRA config "
                        "without loading model weights or training")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
