"""
SFT training on the MATH dataset using Qwen2.5-Math-1.5B.

Trains on data/MATH/sft.jsonl, evaluates on data/MATH/validation.jsonl.
Uses one GPU for the policy model and a second for the vLLM eval instance.

Usage (2x H100):
    # Full dataset
    # effective batch size = 8 * 4 = 32; ~35 GB on policy GPU
    python scripts/train_sft.py \
        --run-name sft_eval1_full \
        --lr 5e-5 --batch-size 8 --grad-accum-steps 4 \
        --num-epochs 5 --eval-interval 30 \
        --policy-device cuda:0 --vllm-device cuda:1

    # Dataset size sweep (more epochs for small datasets)
    for N in 128 256 512 1024; do
        EPOCHS=$([[ $N -le 256 ]] && echo 8 || echo 5)
        python scripts/train_sft.py \
            --run-name sft_eval1_$N \
            --max-examples $N \
            --lr 5e-5 --batch-size 8 --grad-accum-steps 4 \
            --num-epochs $EPOCHS --eval-interval 20 \
            --policy-device cuda:0 --vllm-device cuda:1
    done

    # Filtered (correct answers only, 1408/1767 examples)
    python scripts/train_sft.py \
        --run-name sft_eval1_filtered_correct \
        --filter-correct \
        --lr 5e-5 --batch-size 8 --grad-accum-steps 4 \
        --num-epochs 5 --eval-interval 30 \
        --policy-device cuda:0 --vllm-device cuda:1
"""
from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import typer
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft_helper import (
    get_response_log_probs,
    log_generations,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)

app = typer.Typer()

DATA_DIR = Path(__file__).parent.parent / "data" / "MATH"
PROMPT_FILE = Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"


# ---------------------------------------------------------------------------
# vLLM helpers (provided in assignment spec)
# ---------------------------------------------------------------------------

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm(policy: torch.nn.Module, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def make_val_prompts(val_examples: list[dict], prompt_template: str) -> tuple[list[str], list[str]]:
    prompts = [prompt_template.format(question=ex["problem"]) for ex in val_examples]
    answers = [ex["answer"] for ex in val_examples]
    return prompts, answers


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    policy: torch.nn.Module,
    llm: LLM,
    val_prompts: list[str],
    val_answers: list[str],
    tokenizer,
    eval_sampling_params: SamplingParams,
    eval_step: int,
    examples_seen: int,
    tokens_seen: int,
    output_path: Path | None = None,
    num_log_examples: int = 5,
) -> dict[str, float]:
    policy.eval()
    load_policy_into_vllm(policy, llm)

    result = log_generations(
        vllm_model=llm,
        prompts=val_prompts,
        ground_truths=val_answers,
        reward_fn=r1_zero_reward_fn,
        sampling_params=eval_sampling_params,
        tokenizer=tokenizer,
        label=f"eval_step={eval_step}",
    )
    metrics = result["metrics"]
    accuracy = metrics["n_correct"] / metrics["n_total"]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for rec in result["records"]:
                f.write(json.dumps(rec) + "\n")

    wandb.log({
        "eval/accuracy": accuracy,
        "eval/avg_reward": metrics["avg_reward"],
        "eval/avg_response_length": metrics["avg_response_length"],
        "eval/avg_response_length_correct": metrics["avg_response_length_correct"],
        "eval/avg_response_length_incorrect": metrics["avg_response_length_incorrect"],
        "eval/avg_token_entropy": metrics["avg_token_entropy"],
        "eval_step": eval_step,
        "examples_seen": examples_seen,
        "tokens_seen": tokens_seen,
    })

    # Log a few generation examples to wandb as a table
    records = result["records"][:num_log_examples]
    table = wandb.Table(columns=["prompt", "response", "ground_truth", "reward", "format_reward", "answer_reward"])
    for rec in records:
        table.add_data(
            rec["prompt"][:300],
            rec["response"][:500],
            rec["ground_truth"],
            rec["reward"],
            rec["format_reward"],
            rec["answer_reward"],
        )
    wandb.log({"eval/generations": table, "eval_step": eval_step})

    policy.train()
    return {"accuracy": accuracy, **metrics}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

@app.command()
def main(
    run_name: str = typer.Option("sft_full", "--run-name"),
    model_id: str = typer.Option("Qwen/Qwen2.5-Math-1.5B", "--model-id"),
    sft_data_path: Path = typer.Option(DATA_DIR / "sft.jsonl", "--sft-data"),
    val_data_path: Path = typer.Option(DATA_DIR / "validation.jsonl", "--val-data"),
    max_examples: int = typer.Option(-1, "--max-examples", help="-1 = full dataset"),
    filter_correct: bool = typer.Option(False, "--filter-correct/--no-filter-correct", help="Keep only SFT examples where the response is correct"),
    lr: float = typer.Option(2e-5, "--lr"),
    batch_size: int = typer.Option(8, "--batch-size", help="Microbatch size per step"),
    grad_accum_steps: int = typer.Option(4, "--grad-accum-steps"),
    num_epochs: int = typer.Option(3, "--num-epochs"),
    max_seq_len: int = typer.Option(1536, "--max-seq-len", help="Truncate sequences longer than this"),
    eval_interval: int = typer.Option(50, "--eval-interval", help="Eval every N optimizer steps"),
    eval_examples: int = typer.Option(500, "--eval-examples", help="Val examples to eval on"),
    eval_max_tokens: int = typer.Option(1024, "--eval-max-tokens"),
    policy_device: str = typer.Option("cuda:0", "--policy-device"),
    vllm_device: str = typer.Option("cuda:1", "--vllm-device"),
    seed: int = typer.Option(42, "--seed"),
    wandb_project: str = typer.Option("cs336-sft", "--wandb-project"),
    gpu_memory_utilization: float = typer.Option(0.85, "--gpu-memory-utilization"),
):
    random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------ setup
    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "model_id": model_id,
            "max_examples": max_examples,
            "filter_correct": filter_correct,
            "lr": lr,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "num_epochs": num_epochs,
            "max_seq_len": max_seq_len,
        },
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    wandb.define_metric("examples_seen")
    wandb.define_metric("tokens_seen")

    prompt_template = PROMPT_FILE.read_text()

    # ------------------------------------------------------------------ data
    sft_data = load_jsonl(sft_data_path)
    if filter_correct:
        before = len(sft_data)
        sft_data = [
            ex for ex in sft_data
            if r1_zero_reward_fn(ex["response"], ex["ground_truth"])["answer_reward"] == 1.0
        ]
        typer.echo(f"Filtered to correct examples: {len(sft_data)}/{before} kept.")
        wandb.config.update({"filtered_dataset_size": len(sft_data)})
    if max_examples > 0:
        sft_data = sft_data[:max_examples]
    typer.echo(f"Training on {len(sft_data)} examples.")

    val_data = load_jsonl(val_data_path)
    random.shuffle(val_data)
    val_data_small = val_data[: min(eval_examples, len(val_data))]
    val_data_final = val_data[: min(5000, len(val_data))]
    val_prompts, val_answers = make_val_prompts(val_data_small, prompt_template)
    final_val_prompts, final_val_answers = make_val_prompts(val_data_final, prompt_template)

    # ------------------------------------------------------------------ model
    typer.echo(f"Loading policy on {policy_device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")
    policy = policy.to(policy_device)
    policy.train()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    # ------------------------------------------------------------------ vLLM
    typer.echo(f"Initialising vLLM on {vllm_device} ...")
    llm = init_vllm(model_id, vllm_device, seed=seed, gpu_memory_utilization=gpu_memory_utilization)
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=eval_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=1,
    )

    # ------------------------------------------------------------------ train
    train_step = 0
    eval_step = 0
    examples_seen = 0
    tokens_seen = 0
    final_eval_output_path = Path("outputs") / "eval" / f"{run_name}.jsonl"
    total_optimizer_steps = num_epochs * ((len(sft_data) + batch_size * grad_accum_steps - 1) // (batch_size * grad_accum_steps))
    typer.echo(f"~{total_optimizer_steps} optimizer steps planned.")

    # Eval before any training
    typer.echo("Running initial evaluation ...")
    run_eval(
        policy, llm, val_prompts, val_answers, tokenizer, eval_sampling_params, eval_step,
        examples_seen, tokens_seen,
    )
    eval_step += 1

    for epoch in range(num_epochs):
        random.shuffle(sft_data)
        typer.echo(f"\nEpoch {epoch + 1}/{num_epochs}")

        optimizer.zero_grad()
        microbatch_idx = 0

        for i in range(0, len(sft_data), batch_size):
            micro_batch = sft_data[i: i + batch_size]
            if not micro_batch:
                continue

            prompts = [ex["prompt"] for ex in micro_batch]
            responses = [ex["response"] for ex in micro_batch]

            # Tokenize
            batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = batch["input_ids"].to(policy_device)
            labels = batch["labels"].to(policy_device)
            response_mask = batch["response_mask"].to(policy_device)

            # Truncate to max_seq_len to avoid OOM
            if input_ids.shape[1] > max_seq_len:
                input_ids = input_ids[:, :max_seq_len]
                labels = labels[:, :max_seq_len]
                response_mask = response_mask[:, :max_seq_len]

            # Skip microbatch if no response tokens survive truncation
            if response_mask.sum() == 0:
                continue

            examples_seen += int(input_ids.shape[0])
            tokens_seen += int(input_ids.numel())

            # Forward + backward
            result = get_response_log_probs(policy, input_ids, labels, return_token_entropy=True)
            log_probs = result["log_probs"]
            token_entropy = result["token_entropy"]

            # normalize by number of response tokens for stable loss across lengths
            normalize_constant = float(response_mask.sum())
            loss, _ = sft_microbatch_train_step(
                log_probs, response_mask, grad_accum_steps, normalize_constant
            )

            # Avg entropy over response tokens for logging
            avg_entropy = (token_entropy * response_mask).sum() / normalize_constant

            microbatch_idx += 1

            # Optimizer step every grad_accum_steps microbatches
            if microbatch_idx % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                train_step += 1

                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_token_entropy": avg_entropy.item(),
                    "train/epoch": epoch + 1,
                    "train_step": train_step,
                    "examples_seen": examples_seen,
                    "tokens_seen": tokens_seen,
                })

                if train_step % 10 == 0:
                    typer.echo(
                        f"  step={train_step}  loss={loss.item():.4f}  "
                        f"entropy={avg_entropy.item():.3f}"
                    )

                # Periodic eval
                if train_step % eval_interval == 0:
                    typer.echo(f"  => Running eval at train_step={train_step}")
                    metrics = run_eval(
                        policy, llm, val_prompts, val_answers,
                        tokenizer, eval_sampling_params, eval_step,
                        examples_seen, tokens_seen,
                    )
                    typer.echo(f"     accuracy={metrics['accuracy']:.4f}")
                    eval_step += 1

        # Flush any leftover accumulated gradients at end of epoch
        if microbatch_idx % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_step += 1

    # ------------------------------------------------------------------ final eval
    typer.echo(f"\nFinal evaluation on {len(final_val_answers)} validation examples ...")
    metrics = run_eval(
        policy,
        llm,
        final_val_prompts,
        final_val_answers,
        tokenizer,
        eval_sampling_params,
        eval_step,
        examples_seen,
        tokens_seen,
        final_eval_output_path,
    )
    typer.echo(f"Final accuracy: {metrics['accuracy']:.4f}")
    typer.echo(f"Final eval results saved to {final_eval_output_path}")

    # Save checkpoint
    ckpt_dir = Path("outputs") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    typer.echo(f"Model saved to {ckpt_dir}")

    wandb.finish()


if __name__ == "__main__":
    app()
