"""
Expert Iteration (EI) on the MATH dataset using Qwen2.5-Math-1.5B.

Algorithm (per EI step):
  1. Sample Db questions from train.jsonl
  2. Generate G rollouts per question with vLLM
  3. Keep only correct rollouts (answer_reward == 1.0)
  4. Run SFT on the filtered set for sft_epochs epochs
  5. Sync updated weights back to vLLM
  6. Evaluate on validation set

Usage (2x H100):
    # Baseline: G=4, sft_epochs=1, db_size=512
    python scripts/train_ei.py \
        --run-name ei_G4_e1_db512 \
        --num-rollouts 4 --sft-epochs 1 --db-size 512 \
        --policy-device cuda:0 --vllm-device cuda:1

    # More rollouts
    python scripts/train_ei.py \
        --run-name ei_G8_e1_db512 \
        --num-rollouts 8 --sft-epochs 1 --db-size 512 \
        --policy-device cuda:0 --vllm-device cuda:1

    python scripts/train_ei.py \
        --run-name ei_G16_e1_db512 \
        --num-rollouts 16 --sft-epochs 1 --db-size 512 \
        --policy-device cuda:0 --vllm-device cuda:1

    # More SFT epochs
    python scripts/train_ei.py \
        --run-name ei_G4_e2_db512 \
        --num-rollouts 4 --sft-epochs 2 --db-size 512 \
        --policy-device cuda:0 --vllm-device cuda:1

    # Larger batch
    python scripts/train_ei.py \
        --run-name ei_G4_e1_db1024 \
        --num-rollouts 4 --sft-epochs 1 --db-size 1024 \
        --policy-device cuda:0 --vllm-device cuda:1
"""
from __future__ import annotations

import json
import random
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
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
    log_generations,
)

app = typer.Typer()

DATA_DIR = Path(__file__).parent.parent / "data" / "MATH"
PROMPT_FILE = Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"


# ---------------------------------------------------------------------------
# vLLM helpers (mirrors train_sft.py)
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


def make_prompts(examples: list[dict], prompt_template: str) -> tuple[list[str], list[str]]:
    prompts = [prompt_template.format(question=ex["problem"]) for ex in examples]
    answers = [ex["answer"] for ex in examples]
    return prompts, answers


# ---------------------------------------------------------------------------
# Rollout + filter
# ---------------------------------------------------------------------------

def generate_and_filter(
    llm: LLM,
    prompts: list[str],
    answers: list[str],
    sampling_params: SamplingParams,
) -> tuple[list[dict], dict]:
    """Generate G rollouts per prompt and keep only correct ones.

    Returns:
        sft_examples: list of {"prompt", "response", "ground_truth"}
        stats: rollout statistics for logging
    """
    outputs = llm.generate(prompts, sampling_params)
    G = sampling_params.n

    total = len(prompts) * G
    n_correct = 0
    n_questions_with_correct = 0
    sft_examples = []
    all_entropies = []

    for prompt, answer, output in zip(prompts, answers, outputs):
        question_correct = 0
        for completion in output.outputs:
            response = completion.text
            reward = r1_zero_reward_fn(response, answer)
            # Entropy proxy from vLLM logprobs: -mean(log p(chosen token))
            if completion.logprobs:
                log_probs = torch.tensor(
                    [list(step.values())[0].logprob for step in completion.logprobs],
                    dtype=torch.float32,
                )
                all_entropies.append(float(-log_probs.mean()))

            if reward["answer_reward"] == 1.0:
                sft_examples.append({
                    "prompt": prompt,
                    "response": response,
                    "ground_truth": answer,
                })
                n_correct += 1
                question_correct += 1

        if question_correct > 0:
            n_questions_with_correct += 1

    stats = {
        "rollout/total": total,
        "rollout/n_correct": n_correct,
        "rollout/pass_rate": n_correct / total if total > 0 else 0.0,
        "rollout/questions_with_correct": n_questions_with_correct,
        "rollout/sft_examples": len(sft_examples),
        "rollout/avg_token_entropy": (
            sum(all_entropies) / len(all_entropies) if all_entropies else float("nan")
        ),
    }
    return sft_examples, stats


# ---------------------------------------------------------------------------
# SFT inner loop
# ---------------------------------------------------------------------------

def run_sft_epoch(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sft_examples: list[dict],
    tokenizer,
    microbatch_size: int,
    grad_accum_steps: int,
    max_seq_len: int,
    device: str,
) -> dict:
    """One epoch of SFT on sft_examples. Returns avg loss and avg entropy."""
    policy.train()
    random.shuffle(sft_examples)

    total_loss = 0.0
    total_entropy = 0.0
    n_steps = 0
    microbatch_idx = 0
    optimizer.zero_grad()

    for i in range(0, len(sft_examples), microbatch_size):
        batch = sft_examples[i: i + microbatch_size]
        if not batch:
            continue

        prompts = [ex["prompt"] for ex in batch]
        responses = [ex["response"] for ex in batch]

        tok = tokenize_prompt_and_output(prompts, responses, tokenizer)
        input_ids = tok["input_ids"].to(device)
        labels = tok["labels"].to(device)
        response_mask = tok["response_mask"].to(device)

        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
            labels = labels[:, :max_seq_len]
            response_mask = response_mask[:, :max_seq_len]

        if response_mask.sum() == 0:
            continue

        result = get_response_log_probs(policy, input_ids, labels, return_token_entropy=True)
        log_probs = result["log_probs"]
        token_entropy = result["token_entropy"]

        normalize_constant = float(response_mask.sum())
        loss, _ = sft_microbatch_train_step(
            log_probs, response_mask, grad_accum_steps, normalize_constant
        )
        avg_entropy = float((token_entropy * response_mask).sum() / normalize_constant)

        total_loss += loss.item()
        total_entropy += avg_entropy
        microbatch_idx += 1

        if microbatch_idx % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            n_steps += 1

    # flush remaining gradients
    if microbatch_idx % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        n_steps += 1

    denom = max(microbatch_idx, 1)
    return {
        "sft/avg_loss": total_loss / denom,
        "sft/avg_token_entropy": total_entropy / denom,
    }


# ---------------------------------------------------------------------------
# Validation eval
# ---------------------------------------------------------------------------

def run_eval(
    policy: torch.nn.Module,
    llm: LLM,
    val_prompts: list[str],
    val_answers: list[str],
    tokenizer,
    eval_sampling_params: SamplingParams,
    ei_step: int,
    num_log_examples: int = 5,
) -> dict:
    policy.eval()
    load_policy_into_vllm(policy, llm)

    result = log_generations(
        vllm_model=llm,
        prompts=val_prompts,
        ground_truths=val_answers,
        reward_fn=r1_zero_reward_fn,
        sampling_params=eval_sampling_params,
        tokenizer=tokenizer,
        label=f"ei_step={ei_step}",
    )
    metrics = result["metrics"]
    accuracy = metrics["n_correct"] / metrics["n_total"]

    wandb.log({
        "eval/accuracy": accuracy,
        "eval/avg_reward": metrics["avg_reward"],
        "eval/avg_response_length": metrics["avg_response_length"],
        "eval/avg_token_entropy": metrics["avg_token_entropy"],
        "ei_step": ei_step,
    })

    records = result["records"][:num_log_examples]
    table = wandb.Table(columns=["prompt", "response", "ground_truth", "reward"])
    for rec in records:
        table.add_data(rec["prompt"][:300], rec["response"][:500], rec["ground_truth"], rec["reward"])
    wandb.log({"eval/generations": table, "ei_step": ei_step})

    policy.train()
    return {"accuracy": accuracy, **metrics}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@app.command()
def main(
    run_name: str = typer.Option("ei_run", "--run-name"),
    model_id: str = typer.Option("Qwen/Qwen2.5-Math-1.5B", "--model-id"),
    train_data_path: Path = typer.Option(DATA_DIR / "train.jsonl", "--train-data"),
    val_data_path: Path = typer.Option(DATA_DIR / "validation.jsonl", "--val-data"),
    # EI hyperparameters
    n_ei_steps: int = typer.Option(5, "--n-ei-steps"),
    db_size: int = typer.Option(1024, "--db-size", help="Questions sampled per EI step"),
    num_rollouts: int = typer.Option(4, "--num-rollouts", help="Rollouts G per question"),
    sft_epochs: int = typer.Option(1, "--sft-epochs", help="SFT epochs per EI step"),
    # SFT training
    lr: float = typer.Option(5e-5, "--lr"),
    microbatch_size: int = typer.Option(8, "--microbatch-size"),
    grad_accum_steps: int = typer.Option(4, "--grad-accum-steps"),
    max_seq_len: int = typer.Option(1536, "--max-seq-len"),
    # Rollout sampling
    sampling_temperature: float = typer.Option(0.7, "--sampling-temperature"),
    sampling_max_tokens: int = typer.Option(1024, "--sampling-max-tokens"),
    sampling_min_tokens: int = typer.Option(4, "--sampling-min-tokens"),
    # Eval
    eval_examples: int = typer.Option(500, "--eval-examples"),
    eval_max_tokens: int = typer.Option(1024, "--eval-max-tokens"),
    # Infrastructure
    policy_device: str = typer.Option("cuda:0", "--policy-device"),
    vllm_device: str = typer.Option("cuda:1", "--vllm-device"),
    gpu_memory_utilization: float = typer.Option(0.85, "--gpu-memory-utilization"),
    seed: int = typer.Option(42, "--seed"),
    wandb_project: str = typer.Option("cs336-ei", "--wandb-project"),
):
    random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------ wandb
    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "model_id": model_id,
            "n_ei_steps": n_ei_steps,
            "db_size": db_size,
            "num_rollouts": num_rollouts,
            "sft_epochs": sft_epochs,
            "lr": lr,
            "microbatch_size": microbatch_size,
            "grad_accum_steps": grad_accum_steps,
            "max_seq_len": max_seq_len,
            "sampling_temperature": sampling_temperature,
        },
    )
    wandb.define_metric("ei_step")
    wandb.define_metric("eval/*", step_metric="ei_step")
    wandb.define_metric("rollout/*", step_metric="ei_step")
    wandb.define_metric("sft/*", step_metric="ei_step")

    prompt_template = PROMPT_FILE.read_text()

    # ------------------------------------------------------------------ data
    train_data = load_jsonl(train_data_path)
    typer.echo(f"Loaded {len(train_data)} training examples.")

    val_data = load_jsonl(val_data_path)
    random.shuffle(val_data)
    val_data_small = val_data[: min(eval_examples, len(val_data))]
    val_data_final = val_data[: min(5000, len(val_data))]
    val_prompts, val_answers = make_prompts(val_data_small, prompt_template)
    final_val_prompts, final_val_answers = make_prompts(val_data_final, prompt_template)

    # ------------------------------------------------------------------ model
    typer.echo(f"Loading policy on {policy_device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    policy = policy.to(policy_device)
    policy.train()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    # ------------------------------------------------------------------ vLLM
    typer.echo(f"Initialising vLLM on {vllm_device} ...")
    llm = init_vllm(model_id, vllm_device, seed=seed, gpu_memory_utilization=gpu_memory_utilization)

    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=num_rollouts,
        seed=seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=1,
    )

    eval_sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=eval_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=1,
    )

    # ------------------------------------------------------------------ initial eval
    typer.echo("Running initial evaluation (before any EI) ...")
    metrics = run_eval(policy, llm, val_prompts, val_answers, tokenizer, eval_sampling_params, ei_step=0)
    typer.echo(f"  Initial accuracy: {metrics['accuracy']:.4f}")

    # ------------------------------------------------------------------ EI loop
    for ei_step in range(1, n_ei_steps + 1):
        typer.echo(f"\n{'='*60}")
        typer.echo(f"EI step {ei_step}/{n_ei_steps}")
        typer.echo(f"{'='*60}")

        # 1. Sample Db questions
        batch = random.sample(train_data, min(db_size, len(train_data)))
        prompts, answers = make_prompts(batch, prompt_template)

        # 2. Generate G rollouts and filter correct ones
        typer.echo(f"  Generating {num_rollouts} rollouts x {len(prompts)} questions ...")
        sft_examples, rollout_stats = generate_and_filter(llm, prompts, answers, sampling_params)
        typer.echo(
            f"  Pass rate: {rollout_stats['rollout/pass_rate']:.3f}  "
            f"({rollout_stats['rollout/n_correct']}/{rollout_stats['rollout/total']})  "
            f"SFT examples: {len(sft_examples)}"
        )
        wandb.log({**rollout_stats, "ei_step": ei_step})

        if not sft_examples:
            typer.echo("  No correct rollouts — skipping SFT step.")
            metrics = run_eval(policy, llm, val_prompts, val_answers, tokenizer, eval_sampling_params, ei_step)
            typer.echo(f"  Eval accuracy: {metrics['accuracy']:.4f}")
            continue

        # 3. SFT on filtered rollouts
        typer.echo(f"  Running SFT for {sft_epochs} epoch(s) on {len(sft_examples)} examples ...")
        epoch_metrics_accum: dict[str, float] = {}
        for epoch in range(sft_epochs):
            epoch_metrics = run_sft_epoch(
                policy, optimizer, sft_examples, tokenizer,
                microbatch_size, grad_accum_steps, max_seq_len, policy_device,
            )
            for k, v in epoch_metrics.items():
                epoch_metrics_accum[k] = v  # keep last epoch's values
            typer.echo(
                f"    epoch {epoch+1}/{sft_epochs}  "
                f"loss={epoch_metrics['sft/avg_loss']:.4f}  "
                f"entropy={epoch_metrics['sft/avg_token_entropy']:.3f}"
            )
        wandb.log({**epoch_metrics_accum, "ei_step": ei_step})

        # 4. Sync weights to vLLM and evaluate
        typer.echo("  Evaluating ...")
        metrics = run_eval(policy, llm, val_prompts, val_answers, tokenizer, eval_sampling_params, ei_step)
        typer.echo(f"  Eval accuracy: {metrics['accuracy']:.4f}")

    # ------------------------------------------------------------------ save
    typer.echo(f"\nRunning final evaluation on {len(final_val_answers)} validation examples ...")
    final_metrics = run_eval(
        policy,
        llm,
        final_val_prompts,
        final_val_answers,
        tokenizer,
        eval_sampling_params,
        ei_step=n_ei_steps,
    )
    typer.echo(f"  Final accuracy: {final_metrics['accuracy']:.4f}")

    ckpt_dir = Path("outputs") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    typer.echo(f"\nModel saved to {ckpt_dir}")

    wandb.finish()


if __name__ == "__main__":
    app()
