"""
Run a math baseline evaluation on the MATH dataset using vLLM.

Loads a JSONL dataset (produced by download_math_data.py), formats prompts,
generates responses with vLLM, and reports accuracy using the reward functions
from drgrpo_grader.py.

Usage:
    uv run python scripts/math_baseline.py \\
        --model Qwen/Qwen2.5-Math-1.5B \\
        --data-path data/math/test.jsonl \\
        --prompt-type r1_zero \\
        --output-path outputs/math_baseline_results.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)

app = typer.Typer()

PROMPT_DIR = Path(__file__).parent.parent / "cs336_alignment" / "prompts"

PROMPT_FILES = {
    "r1_zero": PROMPT_DIR / "r1_zero.prompt",
    "question_only": PROMPT_DIR / "question_only.prompt",
}

REWARD_FNS = {
    "r1_zero": r1_zero_reward_fn,
    "question_only": question_only_reward_fn,
}


def load_prompt_template(prompt_type: str) -> str:
    path = PROMPT_FILES[prompt_type]
    return path.read_text()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@app.command()
def main(
    model: str = typer.Option(
        "Qwen/Qwen2.5-Math-1.5B",
        "--model",
        help="HuggingFace model ID or local path.",
    ),
    data_path: Path = typer.Option(
        ...,
        "--data-path",
        help="Path to the JSONL dataset file.",
    ),
    prompt_type: str = typer.Option(
        "r1_zero",
        "--prompt-type",
        help="Prompt type: 'r1_zero' or 'question_only'.",
    ),
    output_path: Path = typer.Option(
        Path("outputs/math_baseline_results.jsonl"),
        "--output-path",
        help="Path to save per-example results as JSONL.",
    ),
    max_examples: int = typer.Option(
        -1,
        "--max-examples",
        help="Maximum number of examples to evaluate (-1 for all).",
    ),
    max_tokens: int = typer.Option(
        2048,
        "--max-tokens",
        help="Maximum number of tokens to generate.",
    ),
    temperature: float = typer.Option(
        0.0,
        "--temperature",
        help="Sampling temperature (0.0 = greedy).",
    ),
    tensor_parallel_size: int = typer.Option(
        1,
        "--tensor-parallel-size",
        help="Number of GPUs to use for tensor parallelism.",
    ),
    fast_grading: bool = typer.Option(
        True,
        "--fast-grading/--no-fast-grading",
        help="Use fast grading (skips math_verify, slightly less accurate).",
    ),
):
    if prompt_type not in PROMPT_FILES:
        typer.echo(f"Unknown prompt type '{prompt_type}'. Choose from: {list(PROMPT_FILES)}")
        raise typer.Exit(1)

    typer.echo(f"Loading data from {data_path} ...")
    examples = load_jsonl(data_path)
    if max_examples > 0:
        examples = examples[:max_examples]
    typer.echo(f"Loaded {len(examples)} examples.")

    prompt_template = load_prompt_template(prompt_type)
    reward_fn = REWARD_FNS[prompt_type]

    prompts = [prompt_template.format(question=ex["problem"]) for ex in examples]

    typer.echo(f"Loading model {model} with vLLM ...")
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["</answer>"] if prompt_type == "r1_zero" else None,
        include_stop_str_in_output=True,
    )

    typer.echo("Generating responses ...")
    outputs = llm.generate(prompts, sampling_params)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(examples)
    correct = 0
    format_correct = 0
    results_by_type: dict[str, dict] = {}

    with open(output_path, "w") as out_f:
        for example, output in zip(examples, outputs):
            response = output.outputs[0].text

            reward_dict = reward_fn(response, example["answer"], fast=fast_grading)
            is_correct = reward_dict["answer_reward"] == 1.0
            has_format = reward_dict["format_reward"] == 1.0

            correct += int(is_correct)
            format_correct += int(has_format)

            ex_type = example.get("type", "unknown")
            if ex_type not in results_by_type:
                results_by_type[ex_type] = {"correct": 0, "total": 0}
            results_by_type[ex_type]["total"] += 1
            results_by_type[ex_type]["correct"] += int(is_correct)

            record = {
                "problem": example["problem"],
                "answer": example["answer"],
                "level": example.get("level", ""),
                "type": ex_type,
                "subset": example.get("subset", ""),
                "response": response,
                "reward": reward_dict["reward"],
                "format_reward": reward_dict["format_reward"],
                "answer_reward": reward_dict["answer_reward"],
            }
            out_f.write(json.dumps(record) + "\n")

    accuracy = correct / total if total > 0 else 0.0
    format_rate = format_correct / total if total > 0 else 0.0

    typer.echo("\n=== Results ===")
    typer.echo(f"Total examples:    {total}")
    typer.echo(f"Accuracy:          {accuracy:.4f} ({correct}/{total})")
    typer.echo(f"Format rate:       {format_rate:.4f} ({format_correct}/{total})")

    typer.echo("\n--- Accuracy by type ---")
    for ex_type, stats in sorted(results_by_type.items()):
        acc = stats["correct"] / stats["total"]
        typer.echo(f"  {ex_type:<35} {acc:.4f} ({stats['correct']}/{stats['total']})")

    typer.echo(f"\nPer-example results saved to {output_path}")


if __name__ == "__main__":
    app()
