"""
Collect zero-shot predictions on AlpacaEval using vLLM (local model)
or a free API backend (e.g. Llama 3.1 8B via Together AI).

Serializes results as a JSON array compatible with alpaca_eval_annotate.py.

Usage (local vLLM):
    python scripts/alpaca_eval_baseline.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --output-path outputs/alpaca_eval_baseline.json

    # With a custom generator name:
    python scripts/alpaca_eval_baseline.py \
        --model outputs/sft/llama-3.1-8b-sft \
        --output-path outputs/alpaca_eval_sft.json \
        --generator llama-3.1-8b-sft

Usage (Together AI API — Llama 3.1 8B):
    python scripts/alpaca_eval_baseline.py \
        --output-path outputs/alpaca_eval_baseline_8b_api.json \
        --generator llama-3.1-8b-instruct \
        --use-api together \
        --api-model meta-llama/Meta-Llama-3-8B-Instruct-Lite
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from cs336_alignment.api_client import BACKEND_CONFIGS, call_chat, load_api_key, make_client

app = typer.Typer()

PROMPT_DIR = Path(__file__).parent.parent / "cs336_alignment" / "prompts"

SYSTEM_PROMPT_TEMPLATE = (PROMPT_DIR / "zero_shot_system_prompt.prompt").read_text()
ALPACA_EVAL_INSTRUCTION_TEMPLATE = (PROMPT_DIR / "alpaca_eval.prompt").read_text()

_CHAT_SYSTEM = (
    "You are a helpful, respectful, and honest assistant. "
    "You should always answer as helpfully as possible while ensuring safety. "
    "Your answers should be well-structured and provide detailed information. "
    "They should also have an engaging tone. "
    "Your responses must not contain any fake, harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content, even if it may be helpful. "
    "Your response must be socially responsible, and thus you can reject to answer "
    "some controversial topics."
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(instruction: str) -> str:
    instruction_str = ALPACA_EVAL_INSTRUCTION_TEMPLATE.format(instruction=instruction)
    return SYSTEM_PROMPT_TEMPLATE.format(instruction=instruction_str)


def format_chat_messages(instruction: str) -> list[dict]:
    instruction_str = ALPACA_EVAL_INSTRUCTION_TEMPLATE.format(instruction=instruction)
    return [
        {"role": "system", "content": _CHAT_SYSTEM},
        {"role": "user", "content": instruction_str},
    ]


@app.command()
def main(
    model: str = typer.Option(
        "",
        "--model",
        help="HuggingFace model ID or local path (vLLM mode). Not needed when using --use-api.",
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
        help="Number of GPUs for tensor parallelism (vLLM mode).",
    ),
    dtype: str = typer.Option(
        "auto",
        "--dtype",
        help="Model dtype: 'auto', 'float16', 'bfloat16' (vLLM mode).",
    ),
    use_api: str = typer.Option(
        "",
        "--use-api",
        help=f"Use a free API backend instead of vLLM. Choices: {list(BACKEND_CONFIGS)}",
    ),
    api_key: str = typer.Option(
        "",
        "--api-key",
        help="API key (defaults to env var for the selected backend).",
    ),
    api_model: str = typer.Option(
        "",
        "--api-model",
        help="Override the model used with --use-api (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct-Lite').",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from existing output file, skipping already-generated examples.",
    ),
):
    typer.echo(f"Loading AlpacaEval examples from {data_path} ...")
    examples = load_jsonl(data_path)
    if max_examples > 0:
        examples = examples[:max_examples]
    typer.echo(f"Loaded {len(examples)} examples.")

    # Resume: load already-done instructions
    done: dict[str, dict] = {}
    if resume and output_path.exists():
        with open(output_path) as f:
            for rec in json.load(f):
                done[rec["instruction"]] = rec
        typer.echo(f"Resuming: {len(done)} already generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_api:
        if use_api not in BACKEND_CONFIGS:
            typer.echo(f"Unknown backend '{use_api}'. Choose from: {list(BACKEND_CONFIGS)}")
            raise typer.Exit(1)

        cfg = BACKEND_CONFIGS[use_api]
        model_name = api_model or cfg["model"]
        rpm = cfg["rpm"]
        seconds_per_request = 60.0 / rpm

        key = load_api_key(use_api, api_key)
        if not key:
            typer.echo(f"No API key found. Set {cfg['env_key']} or pass --api-key.")
            raise typer.Exit(1)

        client = make_client(use_api, key)
        typer.echo(f"Using backend: {use_api} | model: {model_name}")

        remaining = [ex for ex in examples if ex["instruction"] not in done]
        typer.echo(f"To generate: {len(remaining)} | Estimated time: ~{len(remaining) / rpm:.1f} min")

        for i, ex in enumerate(remaining):
            typer.echo(f"[{i+1}/{len(remaining)}] Generating...")
            messages = format_chat_messages(ex["instruction"])
            t0 = time.perf_counter()
            text = call_chat(client, model_name, messages, max_tokens=max_tokens, temperature=0.0)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, seconds_per_request - elapsed))

            done[ex["instruction"]] = {
                "instruction": ex["instruction"],
                "output": text or "",
                "generator": generator,
                "dataset": ex.get("dataset", ""),
            }

            # Write incrementally
            with open(output_path, "w") as f:
                json.dump(list(done.values()), f, indent=2)

    else:
        if not model:
            typer.echo("--model is required when not using --use-api")
            raise typer.Exit(1)

        from vllm import LLM, SamplingParams

        remaining = [ex for ex in examples if ex["instruction"] not in done]
        prompts = [format_prompt(ex["instruction"]) for ex in remaining]

        typer.echo(f"Loading model {model} with vLLM ...")
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype=dtype,
        )

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
        typer.echo(f"Generation time: {elapsed:.1f}s  ({len(remaining)/elapsed:.2f} examples/sec)")

        for ex, output in zip(remaining, outputs):
            done[ex["instruction"]] = {
                "instruction": ex["instruction"],
                "output": output.outputs[0].text,
                "generator": generator,
                "dataset": ex.get("dataset", ""),
            }

        with open(output_path, "w") as f:
            json.dump(list(done.values()), f, indent=2)

    typer.echo(f"\nWrote {len(done)} predictions to {output_path}")


if __name__ == "__main__":
    app()
