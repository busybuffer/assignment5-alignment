"""
Evaluate Llama 3.1 8B zero-shot performance on GSM8K using vLLM.

Loads GSM8K JSONL examples, formats them using the zero-shot system prompt,
generates responses with greedy decoding, parses the last number as the
predicted answer, computes accuracy, and serializes results to disk.

Usage:
    python scripts/gsm8k_baseline.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --data-path data/gsm8k/test.jsonl \
        --output-path outputs/gsm8k_baseline_results.jsonl

    # Limit examples for quick testing:
    python scripts/gsm8k_baseline.py \
        --model meta-llama/Meta-Llama-3.1-8B \
        --data-path data/gsm8k/test.jsonl \
        --output-path outputs/gsm8k_baseline_results.jsonl \
        --max-examples 100

    CUDA_VISIBLE_DEVICES=1 python scripts/gsm8k_baseline.py \
        --model outputs/sft/llama-3.1-8b-sft \
        --data-path data/gsm8k/test.jsonl \
        --output-path outputs/gsm8k_sft_results.jsonl

"""
from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from vllm import LLM, SamplingParams

from cs336_alignment.metrics import parse_gsm8k_response

app = typer.Typer()

PROMPT_DIR = Path(__file__).parent.parent / "cs336_alignment" / "prompts"

SYSTEM_PROMPT_TEMPLATE = (PROMPT_DIR / "zero_shot_system_prompt.prompt").read_text()
GSM8K_INSTRUCTION_TEMPLATE = (PROMPT_DIR / "gsm8k.prompt").read_text()


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_gold_answer(answer_field: str) -> str:
    """Extract the numeric answer after '####' in the GSM8K answer field."""
    return answer_field.split("####")[-1].strip().replace(",", "")


def format_prompt(example: dict) -> str:
    instruction = GSM8K_INSTRUCTION_TEMPLATE.format(question=example["question"])
    return SYSTEM_PROMPT_TEMPLATE.format(instruction=instruction)


@app.command()
def main(
    model: str = typer.Option(
        ...,
        "--model",
        help="HuggingFace model ID or local path to Llama 3.1 8B.",
    ),
    data_path: Path = typer.Option(
        Path("data/gsm8k/test.jsonl"),
        "--data-path",
        help="Path to GSM8K JSONL file.",
    ),
    output_path: Path = typer.Option(
        Path("outputs/gsm8k_baseline_results.jsonl"),
        "--output-path",
        help="Path to write per-example results as JSONL.",
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
    typer.echo(f"Loading GSM8K examples from {data_path} ...")
    examples = load_jsonl(data_path)
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

    with open(output_path, "w") as out_f:
        for example, output in zip(examples, outputs):
            response = output.outputs[0].text
            gold = extract_gold_answer(example["answer"])
            predicted = parse_gsm8k_response(response)
            is_correct = predicted == gold

            total += 1
            correct += int(is_correct)
            if predicted is None:
                unparseable += 1

            record = {
                "question": example["question"],
                "answer": example["answer"],
                "gold": gold,
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
    typer.echo(f"\nPer-example results saved to {output_path}")


if __name__ == "__main__":
    app()



# # Error analysis
# python3 -c "
# import json
# import random
# results = [json.loads(l) for l in open('outputs/gsm8k_baseline_results.jsonl')]
# failures = [r for r in results if r['predicted'] is None or r['predicted'] != r['gold']]
# print(f'Total: {len(results)}, Unparseable: {len(failures)}')
# for r in random.sample(failures, min(10, len(failures))):
#     print('---')
#     print('Question:', r['question'])
#     print('Gold answer:', r['gold'])
#     print('Predicted answer:', r['predicted'])
#     # print('Response:', repr(r['response'][:300]))
# "

