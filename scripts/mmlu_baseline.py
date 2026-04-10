"""
Evaluate Llama 3.1 8B (base or SFT) performance on MMLU using vLLM.

Supports two prompt formats:
  - "zero_shot": zero-shot system prompt (for base model)
  - "sft":       Alpaca instruction-tuning template (for SFT model)

Usage:
    # Base model (zero-shot):
    python scripts/mmlu_baseline.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --data-dir data/mmlu/test \
        --output-path outputs/mmlu_baseline_results.jsonl

    # SFT model:
    python scripts/mmlu_baseline.py \
        --model outputs/sft/llama-3.1-8b-sft \
        --data-dir data/mmlu/test \
        --output-path outputs/mmlu_sft_results.jsonl \
        --prompt-format sft

    # Limit examples for quick testing:
    python scripts/mmlu_baseline.py \
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

# Zero-shot system prompt (base model)
SYSTEM_PROMPT_TEMPLATE = (PROMPT_DIR / "zero_shot_system_prompt.prompt").read_text()

# Alpaca SFT template (instruction-tuned model)
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

# MMLU-specific instruction (shared by both formats)
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


def format_prompt(example: dict, prompt_format: str = "zero_shot") -> str:
    """Format an MMLU example into the full prompt string.

    prompt_format: "zero_shot" (base model) or "sft" (Alpaca-tuned model).
    """
    instruction = MMLU_INSTRUCTION_TEMPLATE.format(
        subject=example["subject"],
        question=example["question"],
        option_a=example["options"][0],
        option_b=example["options"][1],
        option_c=example["options"][2],
        option_d=example["options"][3],
    )
    if prompt_format == "sft":
        return ALPACA_TEMPLATE.format(instruction=instruction)
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
    prompt_format: str = typer.Option(
        "zero_shot",
        "--prompt-format",
        help="Prompt format: 'zero_shot' (base model) or 'sft' (Alpaca-tuned model).",
    ),
):
    if prompt_format not in ("zero_shot", "sft"):
        typer.echo(f"Unknown prompt format '{prompt_format}'. Choose 'zero_shot' or 'sft'.")
        raise typer.Exit(1)

    typer.echo(f"Prompt format: {prompt_format}")
    typer.echo(f"Loading MMLU examples from {data_dir} ...")
    examples = load_mmlu_examples(data_dir)
    if max_examples > 0:
        examples = examples[:max_examples]
    typer.echo(f"Loaded {len(examples)} examples.")

    prompts = [format_prompt(ex, prompt_format) for ex in examples]

    typer.echo(f"Loading model {model} with vLLM ...")
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype=dtype,
    )

    # Greedy decoding. Stop tokens differ by format:
    # - zero_shot: "# Query:" starts the next conversation turn
    # - sft: "###" starts the next Alpaca section header
    stop_tokens = ["###"] if prompt_format == "sft" else ["# Query:"]
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=stop_tokens,
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
                "prompt_format": prompt_format,
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


# # Error analysis
# python3 -c "
# import json
# results = [json.loads(l) for l in open('outputs/mmlu_dpo_results.jsonl')]
# failures = [r for r in results if r['predicted'] is None]
# print(f'Total: {len(results)}, Unparseable: {len(failures)}')
# for r in failures[:5]:
#     print('---')
#     print('Subject:', r['subject'])
#     print('Question:', r['question'])
#     print('Gold answer:', r['answer'])
#     print('Predicted answer:', r['predicted'])
#     print('Response:', repr(r['response'][:300]))
# "

