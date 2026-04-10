"""
Collect Llama 3.1 8B zero-shot predictions on SimpleSafetyTests using vLLM.

Loads SimpleSafetyTests CSV, generates outputs with greedy decoding, and
serializes results as JSONL compatible with scripts/evaluate_safety.py.

Usage:
    uv run python scripts/sst_baseline.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --output-path outputs/sst_baseline_results.jsonl

    # Limit examples for quick testing:
    uv run python scripts/sst_baseline.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --output-path outputs/sst_baseline_results.jsonl \
        --max-examples 20

Then evaluate with:
    uv run python scripts/evaluate_safety.py \
        --input-path outputs/sst_baseline_results.jsonl \
        --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \
        --num-gpus 2 \
        --output-path outputs/sst_baseline_annotated.jsonl
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import typer
from vllm import LLM, SamplingParams

app = typer.Typer()

PROMPT_DIR = Path(__file__).parent.parent / "cs336_alignment" / "prompts"

SYSTEM_PROMPT_TEMPLATE = (PROMPT_DIR / "zero_shot_system_prompt.prompt").read_text()


def load_sst(path: Path) -> list[dict]:
    examples = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(dict(row))
    return examples


def format_prompt(instruction: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(instruction=instruction)


@app.command()
def main(
    model: str = typer.Option(
        ...,
        "--model",
        help="HuggingFace model ID or local path to Llama 3.1 8B.",
    ),
    data_path: Path = typer.Option(
        Path("data/simple_safety_tests/simple_safety_tests.csv"),
        "--data-path",
        help="Path to SimpleSafetyTests CSV file.",
    ),
    output_path: Path = typer.Option(
        Path("outputs/sst_baseline_results.jsonl"),
        "--output-path",
        help="Path to write predictions as JSONL.",
    ),
    max_examples: int = typer.Option(
        -1,
        "--max-examples",
        help="Maximum number of examples to evaluate (-1 for all).",
    ),
    max_tokens: int = typer.Option(
        512,
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
    typer.echo(f"Loading SimpleSafetyTests from {data_path} ...")
    examples = load_sst(data_path)
    if max_examples > 0:
        examples = examples[:max_examples]
    typer.echo(f"Loaded {len(examples)} examples.")

    prompts = [format_prompt(ex["prompts_final"]) for ex in examples]

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
    with open(output_path, "w") as out_f:
        for example, output in zip(examples, outputs):
            record = {
                **example,
                "output": output.outputs[0].text,
            }
            out_f.write(json.dumps(record) + "\n")

    typer.echo(f"\nWrote {len(examples)} predictions to {output_path}")
    typer.echo("Run safety evaluation with:")
    typer.echo(
        f"  uv run python scripts/evaluate_safety.py "
        f"--input-path {output_path} "
        f"--model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct "
        f"--num-gpus 2 "
        f"--output-path outputs/sst_baseline_annotated.jsonl"
    )


if __name__ == "__main__":
    app()
