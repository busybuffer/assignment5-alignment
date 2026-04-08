"""
Leaderboard validation script for grpo_prompt_ablation / leaderboard problems.

Loads a saved checkpoint and evaluates on the full MATH validation set (5000 examples)
using the exact leaderboard constraints:
  - R1-Zero prompt
  - temperature=1.0, max_tokens=1024
  - r1_zero_reward_fn reward function
  - All 5000 validation examples

Usage:
    python scripts/validate_leaderboard.py \
        --checkpoint outputs/grpo_leaderboard/checkpoint \
        --vllm-device cuda:1

    # Evaluate a mid-training checkpoint (step 50):
    python scripts/validate_leaderboard.py \
        --checkpoint outputs/grpo_leaderboard/checkpoint_step50 \
        --vllm-device cuda:1
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch
import typer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft_helper import log_generations

app = typer.Typer()

DATA_DIR = Path(__file__).parent.parent / "data" / "MATH"
PROMPT_FILE = Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"


@app.command()
def main(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to saved model checkpoint directory."),
    val_data_path: Path = typer.Option(DATA_DIR / "validation.jsonl", "--val-data"),
    val_examples: int = typer.Option(5000, "--val-examples", help="Number of val examples (5000 = full set)."),
    vllm_device: str = typer.Option("cuda:0", "--vllm-device"),
    gpu_memory_utilization: float = typer.Option(0.9, "--gpu-memory-utilization"),
    seed: int = typer.Option(42, "--seed"),
    output_file: Path = typer.Option(None, "--output-file", help="Optional path to save results as JSON."),
):
    typer.echo(f"Loading validation data from {val_data_path} ...")
    with open(val_data_path) as f:
        val_data = [json.loads(l) for l in f if l.strip()]
    val_data = val_data[:val_examples]
    typer.echo(f"  {len(val_data)} examples loaded.")

    prompt_template = PROMPT_FILE.read_text()
    val_prompts = [prompt_template.format(question=ex["problem"]) for ex in val_data]
    val_answers = [ex["answer"] for ex in val_data]

    typer.echo(f"Loading model from {checkpoint} on {vllm_device} ...")
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        llm = LLM(
            model=str(checkpoint),
            device=vllm_device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    sampling_params = SamplingParams(
        temperature=1.0,       # leaderboard requirement
        max_tokens=1024,       # leaderboard requirement
        min_tokens=4,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    typer.echo("Running evaluation ...")
    result = log_generations(
        vllm_model=llm,
        prompts=val_prompts,
        ground_truths=val_answers,
        reward_fn=r1_zero_reward_fn,
        sampling_params=sampling_params,
        tokenizer=None,
        label="leaderboard_eval",
    )

    metrics = result["metrics"]
    records = result["records"]
    accuracy = metrics["n_correct"] / metrics["n_total"]
    avg_answer_reward = sum(r["answer_reward"] for r in records) / len(records)
    avg_format_reward = sum(r["format_reward"] for r in records) / len(records)

    typer.echo("\n===== Leaderboard Results =====")
    typer.echo(f"  Accuracy (answer reward):  {accuracy:.4f}")
    typer.echo(f"  Avg answer reward:         {avg_answer_reward:.4f}")
    typer.echo(f"  Avg format reward:         {avg_format_reward:.4f}")
    typer.echo(f"  Avg reward:                {metrics['avg_reward']:.4f}")
    typer.echo(f"  Avg response length:       {metrics['avg_response_length']:.1f}")
    typer.echo(f"  N correct / N total:       {metrics['n_correct']} / {metrics['n_total']}")
    typer.echo("================================")

    if output_file is not None:
        output = {
            "checkpoint": str(checkpoint),
            "val_examples": len(records),
            "accuracy": accuracy,
            "avg_answer_reward": avg_answer_reward,
            "avg_format_reward": avg_format_reward,
            "avg_reward": metrics["avg_reward"],
            "avg_response_length": metrics["avg_response_length"],
            "n_correct": metrics["n_correct"],
            "n_total": metrics["n_total"],
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        typer.echo(f"Results saved to {output_file}")


if __name__ == "__main__":
    app()
