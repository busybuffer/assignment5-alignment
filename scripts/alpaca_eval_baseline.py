"""
Collect Llama 3.1 8B zero-shot predictions on AlpacaEval using vLLM.

Loads AlpacaEval instructions, generates outputs with greedy decoding, and
serializes results as a JSON array compatible with the AlpacaEval evaluator.

Usage:
    uv run python scripts/alpaca_eval_baseline.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --output-path outputs/alpaca_eval_baseline.json

    # With a custom generator name:
    uv run python scripts/alpaca_eval_baseline.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --output-path outputs/alpaca_eval_baseline.json \
        --generator llama-3.1-8b-base
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from vllm import LLM, SamplingParams

app = typer.Typer()

PROMPT_DIR = Path(__file__).parent.parent / "cs336_alignment" / "prompts"

SYSTEM_PROMPT_TEMPLATE = (PROMPT_DIR / "zero_shot_system_prompt.prompt").read_text()
ALPACA_EVAL_INSTRUCTION_TEMPLATE = (PROMPT_DIR / "alpaca_eval.prompt").read_text()


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(instruction: str) -> str:
    instruction_str = ALPACA_EVAL_INSTRUCTION_TEMPLATE.format(instruction=instruction)
    return SYSTEM_PROMPT_TEMPLATE.format(instruction=instruction_str)


@app.command()
def main(
    model: str = typer.Option(
        ...,
        "--model",
        help="HuggingFace model ID or local path to Llama 3.1 8B.",
    ),
    data_path: Path = typer.Option(
        Path("data/alpaca_eval/alpaca_eval.jsonl"),
        "--data-path",
        help="Path to AlpacaEval JSONL file.",
    ),
    output_path: Path = typer.Option(
        Path("outputs/alpaca_eval_baseline.json"),
        "--output-path",
        help="Path to write predictions as a JSON array.",
    ),
    generator: str = typer.Option(
        "llama-3.1-8b-base",
        "--generator",
        help="Generator identifier stored in each output entry.",
    ),
    max_examples: int = typer.Option(
        -1,
        "--max-examples",
        help="Maximum number of examples to evaluate (-1 for all).",
    ),
    max_tokens: int = typer.Option(
        2048,
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
    typer.echo(f"Loading AlpacaEval examples from {data_path} ...")
    examples = load_jsonl(data_path)
    if max_examples > 0:
        examples = examples[:max_examples]
    typer.echo(f"Loaded {len(examples)} examples.")

    prompts = [format_prompt(ex["instruction"]) for ex in examples]

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

    results = []
    for example, output in zip(examples, outputs):
        results.append({
            "instruction": example["instruction"],
            "output": output.outputs[0].text,
            "generator": generator,
            "dataset": example["dataset"],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    typer.echo(f"\nWrote {len(results)} predictions to {output_path}")
    typer.echo(f"Run evaluation with:")
    typer.echo(
        f"  uv run alpaca_eval --model_outputs {output_path} "
        f"--annotators_config 'scripts/alpaca_eval_vllm_llama3_3_70b_fn' "
        f"--base-dir '.'"
    )


if __name__ == "__main__":
    app()
