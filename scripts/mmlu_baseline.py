"""
Evaluate Llama 3.1 8B zero-shot performance on MMLU using vLLM.

Loads MMLU CSV examples, formats them using the zero-shot system prompt,
generates responses with greedy decoding, parses predicted answer letters,
computes accuracy (overall and per subject), and serializes results to disk.

Usage:
    uv run python scripts/mmlu_baseline.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --data-dir data/mmlu/test \
        --output-path outputs/mmlu_baseline_results.jsonl

    # Limit examples for quick testing:
    uv run python scripts/mmlu_baseline.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --data-dir data/mmlu/test \
        --output-path outputs/mmlu_baseline_results.jsonl \
        --max-examples 100
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import typer
from vllm import LLM, SamplingParams

from cs336_alignment.metrics import parse_mmlu_response

app = typer.Typer()

PROMPT_DIR = Path(__file__).parent.parent / "cs336_alignment" / "prompts"

# The zero-shot system prompt wraps {instruction} inside a Query/Answer block.
SYSTEM_PROMPT_TEMPLATE = (PROMPT_DIR / "zero_shot_system_prompt.prompt").read_text()

# MMLU-specific instruction injected as {instruction} into the system prompt.
MMLU_INSTRUCTION_TEMPLATE = (PROMPT_DIR / "mmlu.prompt").read_text()

LETTER_MAP = ["A", "B", "C", "D"]


def subject_from_path(csv_path: Path) -> str:
    """Convert filename like 'high_school_geography_test.csv' → 'high school geography'."""
    stem = csv_path.stem  # e.g. 'high_school_geography_test'
    # Remove trailing _test or _dev
    for suffix in ("_test", "_dev", "_val"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.replace("_", " ")


def load_mmlu_examples(data_dir: Path) -> list[dict]:
    """Load all MMLU CSV files from data_dir into a list of example dicts."""
    examples = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        subject = subject_from_path(csv_path)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                question, a, b, c, d, answer = row[0], row[1], row[2], row[3], row[4], row[5]
                examples.append(
                    {
                        "subject": subject,
                        "question": question,
                        "options": [a, b, c, d],
                        "answer": answer.strip().upper(),
                    }
                )
    return examples


def format_prompt(example: dict) -> str:
    """Format an MMLU example into the full prompt string."""
    instruction = MMLU_INSTRUCTION_TEMPLATE.format(
        subject=example["subject"],
        question=example["question"],
        option_a=example["options"][0],
        option_b=example["options"][1],
        option_c=example["options"][2],
        option_d=example["options"][3],
    )
    return SYSTEM_PROMPT_TEMPLATE.format(instruction=instruction)


@app.command()
def main(
    model: str = typer.Option(
        ...,
        "--model",
        help="HuggingFace model ID or local path to Llama 3.1 8B.",
    ),
    data_dir: Path = typer.Option(
        Path("data/mmlu/test"),
        "--data-dir",
        help="Directory containing MMLU CSV files.",
    ),
    output_path: Path = typer.Option(
        Path("outputs/mmlu_baseline_results.jsonl"),
        "--output-path",
        help="Path to write per-example results as JSONL.",
    ),
    max_examples: int = typer.Option(
        -1,
        "--max-examples",
        help="Maximum number of examples to evaluate (-1 for all).",
    ),
    max_tokens: int = typer.Option(
        128,
        "--max-tokens",
        help="Maximum number of tokens to generate per example.",
    ),
    tensor_parallel_size: int = typer.Option(
        1,
        "--tensor-parallel-size",
        help="Number of GPUs for tensor parallelism.",
    ),
    dtype: str = typer.Option(
        "auto",
        "--dtype",
        help="Model dtype: 'auto', 'float16', 'bfloat16'.",
    ),
):
    typer.echo(f"Loading MMLU examples from {data_dir} ...")
    examples = load_mmlu_examples(data_dir)
    if max_examples > 0:
        examples = examples[:max_examples]
    typer.echo(f"Loaded {len(examples)} examples.")

    prompts = [format_prompt(ex) for ex in examples]

    typer.echo(f"Loading model {model} with vLLM ...")
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype=dtype,
    )

    # Greedy decoding; stop when the model starts the next conversation turn.
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["# Query:"],
    )

    typer.echo("Generating responses ...")
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0
    throughput = len(examples) / elapsed
    typer.echo(f"Generation time: {elapsed:.1f}s  ({throughput:.2f} examples/sec)")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    unparseable = 0
    by_subject: dict[str, dict] = {}

    with open(output_path, "w") as out_f:
        for example, output in zip(examples, outputs):
            response = output.outputs[0].text

            predicted = parse_mmlu_response(example, response)
            gold = example["answer"]
            is_correct = predicted == gold

            total += 1
            correct += int(is_correct)
            if predicted is None:
                unparseable += 1

            subj = example["subject"]
            if subj not in by_subject:
                by_subject[subj] = {"correct": 0, "total": 0}
            by_subject[subj]["total"] += 1
            by_subject[subj]["correct"] += int(is_correct)

            record = {
                "subject": subj,
                "question": example["question"],
                "options": example["options"],
                "answer": gold,
                "prompt": prompts[total - 1],
                "response": response,
                "predicted": predicted,
                "correct": is_correct,
            }
            out_f.write(json.dumps(record) + "\n")

    accuracy = correct / total if total > 0 else 0.0

    typer.echo("\n=== Results ===")
    typer.echo(f"Total examples:   {total}")
    typer.echo(f"Correct:          {correct}")
    typer.echo(f"Accuracy:         {accuracy:.4f} ({correct}/{total})")
    typer.echo(f"Unparseable:      {unparseable}")

    typer.echo("\n--- Accuracy by subject ---")
    for subj, stats in sorted(by_subject.items()):
        acc = stats["correct"] / stats["total"]
        typer.echo(f"  {subj:<45} {acc:.4f} ({stats['correct']}/{stats['total']})")

    typer.echo(f"\nPer-example results saved to {output_path}")


if __name__ == "__main__":
    app()
