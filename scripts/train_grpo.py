"""
GRPO training on MATH (Qwen2.5-Math-1.5B) with vLLM rollouts.

Uses the ``r1_zero`` prompt and stops generation at ``</answer>`` (see ``SamplingParams``).
Validation metrics are logged to Weights & Biases (including ``eval/accuracy``,
``eval/avg_answer_reward`` for answer-only signal, ``eval/avg_format_reward``, ``eval/avg_reward``).

Learning-rate sweep (one run per job; use the same ``--wandb-group`` to compare curves):

    bash scripts/grpo_lr_sweep.sh

Running (from repo root, after ``uv sync`` and ``wandb login``):

    # Default hyperparameters, 2 GPUs (policy + vLLM)
     python scripts/train_grpo.py \
        --run-name grpo_rwb \
        --val-examples 1024 \
        --policy-device cuda:0 --vllm-device cuda:1

    # No baseline: optimize with per-rollout raw reward from the grader
     python scripts/train_grpo.py \
        --run-name grpo_no_bl \
        --loss-type no_baseline \
        --policy-device cuda:0 --vllm-device cuda:1

    # Length normalization comparison (grpo_length_normalization):
    #   Run A: masked_mean (per-token average, default)
     python scripts/train_grpo.py \
        --run-name grpo_lengnorm_mean \
        --length-norm masked_mean \
        --wandb-group grpo_lengnorm_sweep \
        --policy-device cuda:0 --vllm-device cuda:1

    #   Run B: masked_normalize (per-sequence sum, no length penalty)
     python scripts/train_grpo.py \
        --run-name grpo_lengnorm_normalize \
        --length-norm masked_normalize \
        --wandb-group grpo_lengnorm_sweep \
        --policy-device cuda:0 --vllm-device cuda:1

    # Std normalization comparison (grpo_group_standard_deviation):
     python scripts/train_grpo.py \
        --run-name grpo_no_std_norm \
        --no-std-normalization \
        --wandb-group grpo_std_sweep \
        --policy-device cuda:0 --vllm-device cuda:1

    # Off-policy GRPO-Clip (grpo_off_policy):
     python scripts/train_grpo.py \
        --run-name grpo_clip_offp \
        --loss-type grpo_clip \
        --epochs-per-rollout-batch 2 \
        --wandb-group grpo_offpolicy_sweep \
        --policy-device cuda:0 --vllm-device cuda:1

    # Off-policy hyperparameter sweep (grpo_off_policy_sweep):
    #   Phase 1 — broad sweep, 50 steps (early-stop bad configs)
    bash scripts/grpo_offpolicy_sweep.sh --n-grpo-steps 50
    #   Phase 2 — focused sweep, 200 steps (best 2-3 configs from phase 1)
    CONFIGS="2x128 4x128" bash scripts/grpo_offpolicy_sweep.sh --n-grpo-steps 200

    # Off-policy GRPO-No-Clip ablation (grpo_off_policy_clip_ablation):
    #   Use best off-policy config (ep2, tb256) with unclipped IS-weighted loss
     python scripts/train_grpo.py \
        --run-name grpo_no_clip_offp \
        --loss-type grpo_no_clip \
        --epochs-per-rollout-batch 2 \
        --wandb-group grpo_clip_ablation \
        --policy-device cuda:0 --vllm-device cuda:1

    # Prompt ablation (grpo_prompt_ablation):
    #   Run A: R1-Zero prompt (baseline)
     python scripts/train_grpo.py \
        --run-name grpo_prompt_r1zero \
        --wandb-group grpo_prompt_ablation \
        --policy-device cuda:0 --vllm-device cuda:1

    #   Run B: question-only prompt + question_only reward fn
     python scripts/train_grpo.py \
        --run-name grpo_prompt_qonly \
        --prompt-type question_only \
        --wandb-group grpo_prompt_ablation \
        --policy-device cuda:0 --vllm-device cuda:1
        
    # Leaderboard run (leaderboard):
    #   Constraints: R1-Zero prompt, eval temperature=1.0 (hardcoded)
    
    # on-policy, best lr 3e-5, 200 steps, no std normalization, no length norm
     python scripts/train_grpo.py \
        --run-name grpo_leaderboard \
        --wandb-group grpo_leaderboard \
        --n-grpo-steps 200 \
        --no-std-normalization \
        --policy-device cuda:0 --vllm-device cuda:1
        
    # off-policy
     python scripts/train_grpo.py \
        --run-name grpo_leaderboard_offp \
        --loss-type grpo_clip \
        --epochs-per-rollout-batch 2 \
        --no-std-normalization \
        --n-grpo-steps 200 \
        --wandb-group grpo_offp_no_std_norm \
        --policy-device cuda:0 --vllm-device cuda:1

"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import torch
import typer
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from cs336_alignment.sft_helper import get_response_log_probs, log_generations, tokenize_prompt_and_output

app = typer.Typer()

DATA_DIR = Path(__file__).parent.parent / "data" / "MATH"
PROMPTS_DIR = Path(__file__).parent.parent / "cs336_alignment" / "prompts"
DEFAULT_PROMPT_FILE = PROMPTS_DIR / "r1_zero.prompt"


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


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def make_prompts(examples: list[dict], prompt_template: str) -> tuple[list[str], list[str]]:
    prompts = [prompt_template.format(question=ex["problem"]) for ex in examples]
    answers = [ex["answer"] for ex in examples]
    return prompts, answers


def rollout_batch(
    llm: LLM,
    prompts: list[str],
    answers: list[str],
    sampling_params: SamplingParams,
) -> tuple[list[str], list[str]]:
    """One vLLM call per prompt with ``n=group_size``; flat lists length rollout_batch_size."""
    outputs = llm.generate(prompts, sampling_params)
    responses: list[str] = []
    repeated_ground_truths: list[str] = []
    for answer, output in zip(answers, outputs):
        for completion in output.outputs:
            responses.append(completion.text)
            repeated_ground_truths.append(answer)
    return responses, repeated_ground_truths


def rollout_train_reward_stats(responses: list[str], ground_truths: list[str], reward_fn=r1_zero_reward_fn) -> dict[str, float]:
    total_r = fmt_r = ans_r = 0.0
    n = len(responses)
    for resp, gt in zip(responses, ground_truths):
        d = reward_fn(resp, gt)
        total_r += d["reward"]
        fmt_r += d["format_reward"]
        ans_r += d["answer_reward"]
    return {
        "train/reward_mean": total_r / n,
        "train/format_reward_mean": fmt_r / n,
        "train/answer_reward_mean": ans_r / n,
    }


def run_validation(
    policy: torch.nn.Module,
    llm: LLM,
    val_prompts: list[str],
    val_answers: list[str],
    tokenizer,
    eval_sampling_params: SamplingParams,
    grpo_step: int,
    examples_seen: int,
    tokens_seen: int,
    reward_fn=r1_zero_reward_fn,
    output_path: Path | None = None,
    num_table_rows: int = 5,
) -> dict:
    policy.eval()
    load_policy_into_vllm(policy, llm)
    result = log_generations(
        vllm_model=llm,
        prompts=val_prompts,
        ground_truths=val_answers,
        reward_fn=reward_fn,
        sampling_params=eval_sampling_params,
        tokenizer=tokenizer,
        label=f"grpo_step={grpo_step}",
    )
    metrics = result["metrics"]
    records = result["records"]
    accuracy = metrics["n_correct"] / metrics["n_total"]
    n_val = len(records)
    avg_answer_reward = sum(r["answer_reward"] for r in records) / n_val
    avg_format_reward = sum(r["format_reward"] for r in records) / n_val

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    wandb.log({
        "eval/accuracy": accuracy,
        "eval/avg_reward": metrics["avg_reward"],
        "eval/avg_answer_reward": avg_answer_reward,
        "eval/avg_format_reward": avg_format_reward,
        "eval/avg_response_length": metrics["avg_response_length"],
        "eval/avg_token_entropy": metrics["avg_token_entropy"],
        "grpo_step": grpo_step,
        "examples_seen": examples_seen,
        "tokens_seen": tokens_seen,
    })
    table = wandb.Table(columns=["prompt", "response", "ground_truth", "reward"])
    for rec in result["records"][:num_table_rows]:
        table.add_data(rec["prompt"][:300], rec["response"][:500], rec["ground_truth"], rec["reward"])
    wandb.log({"eval/generations": table, "grpo_step": grpo_step})
    policy.train()
    return {"accuracy": accuracy, "metrics": metrics, "records": records}


@app.command()
def main(
    run_name: str = typer.Option("grpo_run", "--run-name"),
    model_id: str = typer.Option("Qwen/Qwen2.5-Math-1.5B", "--model-id"),
    train_data_path: Path = typer.Option(DATA_DIR / "train.jsonl", "--train-data"),
    val_data_path: Path = typer.Option(DATA_DIR / "validation.jsonl", "--val-data"),
    prompt_type: str | None = typer.Option(None, "--prompt-type", help="r1_zero | question_only  (sets both --prompt-file and --reward-fn automatically)"),
    prompt_file: Path = typer.Option(DEFAULT_PROMPT_FILE, "--prompt-file", help="Path to .prompt template file (use {question} placeholder). Overridden by --prompt-type."),
    reward_fn_name: str = typer.Option("r1_zero", "--reward-fn", help="r1_zero | question_only. Overridden by --prompt-type."),
    n_grpo_steps: int = typer.Option(50, "--n-grpo-steps"),
    learning_rate: float = typer.Option(3e-5, "--learning-rate"),
    advantage_eps: float = typer.Option(1e-6, "--advantage-eps"),
    rollout_batch_size: int = typer.Option(256, "--rollout-batch-size"),
    group_size: int = typer.Option(8, "--group-size"),
    sampling_temperature: float = typer.Option(1.0, "--sampling-temperature"),
    sampling_min_tokens: int = typer.Option(4, "--sampling-min-tokens"),
    sampling_max_tokens: int = typer.Option(1024, "--sampling-max-tokens"),
    epochs_per_rollout_batch: int = typer.Option(1, "--epochs-per-rollout-batch"),
    train_batch_size: int = typer.Option(256, "--train-batch-size"),
    gradient_accumulation_steps: int = typer.Option(64, "--gradient-accumulation-steps"),
    gpu_memory_utilization: float = typer.Option(0.85, "--gpu-memory-utilization"),
    loss_type: str = typer.Option(
        "reinforce_with_baseline",
        "--loss-type",
        help="no_baseline | reinforce_with_baseline | grpo_clip (off-policy)",
    ),
    use_std_normalization: bool = typer.Option(True, "--use-std-normalization/--no-std-normalization"),
    length_norm: str = typer.Option(
        "masked_mean",
        "--length-norm",
        help="masked_mean (per-token avg) | masked_normalize (per-sequence sum, no length penalty).",
    ),
    cliprange: float = typer.Option(0.2, "--cliprange", help="PPO-style clip for grpo_clip."),
    max_seq_len: int = typer.Option(2048, "--max-seq-len"),
    val_examples: int = typer.Option(1024, "--val-examples", help=">=1024 recommended."),
    val_every: int = typer.Option(5, "--val-every"),
    policy_device: str = typer.Option("cuda:0", "--policy-device"),
    vllm_device: str = typer.Option("cuda:1", "--vllm-device"),
    seed: int = typer.Option(42, "--seed"),
    wandb_project: str = typer.Option("cs336-grpo", "--wandb-project"),
    wandb_group: str | None = typer.Option(
        None,
        "--wandb-group",
        help="Optional W&B group name so multiple runs appear together (e.g. LR sweeps).",
    ),
    wandb_tags: str = typer.Option(
        "",
        "--wandb-tags",
        help="Comma-separated W&B tags (e.g. lr_sweep,1e-5).",
    ),
):
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be >= group_size"
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    if epochs_per_rollout_batch == 1:
        assert train_batch_size == rollout_batch_size, (
            "On-policy (epochs_per_rollout_batch=1): set train_batch_size == rollout_batch_size "
            f"(got {train_batch_size} vs {rollout_batch_size})"
        )

    if prompt_type is not None:
        if prompt_type == "r1_zero":
            prompt_file = PROMPTS_DIR / "r1_zero.prompt"
            reward_fn_name = "r1_zero"
        elif prompt_type == "question_only":
            prompt_file = PROMPTS_DIR / "question_only.prompt"
            reward_fn_name = "question_only"
        else:
            raise typer.BadParameter(f"Unknown prompt_type: {prompt_type}. Choose r1_zero | question_only")

    if loss_type not in ("no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"):
        raise typer.BadParameter(f"Unknown loss_type: {loss_type}")

    if reward_fn_name == "r1_zero":
        reward_fn = r1_zero_reward_fn
    elif reward_fn_name == "question_only":
        reward_fn = question_only_reward_fn
    else:
        raise typer.BadParameter(f"Unknown reward_fn: {reward_fn_name}")

    random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path("outputs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_log_path = out_dir / "rollout_samples.jsonl"

    tags = [t.strip() for t in wandb_tags.split(",") if t.strip()]
    wandb.init(
        project=wandb_project,
        name=run_name,
        group=wandb_group,
        tags=tags or None,
        config={
            "model_id": model_id,
            "n_grpo_steps": n_grpo_steps,
            "learning_rate": learning_rate,
            "advantage_eps": advantage_eps,
            "rollout_batch_size": rollout_batch_size,
            "group_size": group_size,
            "sampling_temperature": sampling_temperature,
            "sampling_min_tokens": sampling_min_tokens,
            "sampling_max_tokens": sampling_max_tokens,
            "epochs_per_rollout_batch": epochs_per_rollout_batch,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "micro_train_batch_size": micro_train_batch_size,
            "n_microbatches_per_rollout_batch": n_microbatches_per_rollout_batch,
            "gpu_memory_utilization": gpu_memory_utilization,
            "loss_type": loss_type,
            "length_norm": length_norm,
            "use_std_normalization": use_std_normalization,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "betas": (0.9, 0.95),
            "grad_clip_norm": 1.0,
        },
    )
    wandb.define_metric("grpo_step")
    wandb.define_metric("eval/*", step_metric="grpo_step")
    wandb.define_metric("train/*", step_metric="grpo_step")
    wandb.define_metric("rollout/*", step_metric="grpo_step")
    wandb.define_metric("examples_seen")
    wandb.define_metric("tokens_seen")

    prompt_template = prompt_file.read_text()
    train_data = load_jsonl(train_data_path)
    val_data = load_jsonl(val_data_path)
    random.shuffle(val_data)
    val_data_small = val_data[: min(val_examples, len(val_data))]
    val_data_final = val_data[: min(5000, len(val_data))]
    val_prompts, val_answers = make_prompts(val_data_small, prompt_template)
    final_val_prompts, final_val_answers = make_prompts(val_data_final, prompt_template)

    typer.echo(f"Loading policy on {policy_device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    policy = policy.to(policy_device)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    typer.echo(f"Initialising vLLM on {vllm_device} ...")
    llm = init_vllm(model_id, vllm_device, seed=seed, gpu_memory_utilization=gpu_memory_utilization)

    rollout_sampling = SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        seed=seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=1,
    )
    eval_sampling = SamplingParams(
        temperature=1.0,  # leaderboard requires temperature=1.0
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=1,
    )

    examples_seen = 0
    tokens_seen = 0
    final_eval_output_path = Path("outputs") / "eval" / f"{run_name}.jsonl"
    typer.echo("Initial validation ...")
    val0 = run_validation(
        policy, llm, val_prompts, val_answers, tokenizer, eval_sampling, 0,
        examples_seen, tokens_seen, reward_fn,
    )
    typer.echo(f"  eval avg_reward={val0['metrics']['avg_reward']:.4f}  acc={val0['accuracy']:.4f}")

    for grpo_step in range(1, n_grpo_steps + 1):
        typer.echo(f"\n{'='*60}\nGRPO step {grpo_step}/{n_grpo_steps}\n{'='*60}")

        batch = random.sample(train_data, min(n_prompts_per_rollout_batch, len(train_data)))
        prompts, answers = make_prompts(batch, prompt_template)
        typer.echo(f"  Rollout: {len(prompts)} prompts × {group_size} samples ...")
        responses, repeated_gt = rollout_batch(llm, prompts, answers, rollout_sampling)
        b_sz = len(responses)
        if b_sz % micro_train_batch_size != 0:
            raise RuntimeError(
                f"Rollout batch size {b_sz} must be divisible by micro_train_batch_size "
                f"{micro_train_batch_size} (got {len(prompts)} prompts × {group_size})."
            )

        advantages, raw_rewards, _ = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=repeated_gt,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        prompts_expanded: list[str] = []
        for p in prompts:
            prompts_expanded.extend([p] * group_size)

        tok = tokenize_prompt_and_output(prompts_expanded, responses, tokenizer)
        input_ids = tok["input_ids"].to(policy_device)
        labels = tok["labels"].to(policy_device)
        response_mask = tok["response_mask"].to(policy_device)

        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
            labels = labels[:, :max_seq_len]
            response_mask = response_mask[:, :max_seq_len]

        reward_stats = rollout_train_reward_stats(responses, repeated_gt, reward_fn)
        wandb.log({
            **reward_stats,
            "grpo_step": grpo_step,
            "rollout/batch_size": len(responses),
            "examples_seen": examples_seen,
            "tokens_seen": tokens_seen,
        })

        dtype = next(policy.parameters()).dtype
        raw_b = raw_rewards.unsqueeze(-1).to(device=policy_device, dtype=dtype)
        adv_b = advantages.unsqueeze(-1).to(device=policy_device, dtype=dtype)

        loss_literal = cast(
            Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
            loss_type,
        )

        old_full: torch.Tensor | None = None
        if loss_type in ("grpo_clip", "grpo_no_clip"):
            policy.eval()
            old_chunks: list[torch.Tensor] = []
            with torch.inference_mode():
                for s in range(0, b_sz, micro_train_batch_size):
                    sl = slice(s, s + micro_train_batch_size)
                    old_chunks.append(
                        get_response_log_probs(policy, input_ids[sl], labels[sl], False)["log_probs"]
                    )
            old_full = torch.cat(old_chunks, dim=0)
            policy.train()

        optimizer.zero_grad(set_to_none=True)
        mb_total = 0
        accum_loss = 0.0
        accum_clip = 0.0
        accum_ent = 0.0
        n_accum_logs = 0

        for _epoch in range(epochs_per_rollout_batch):
            for s in range(0, b_sz, micro_train_batch_size):
                sl = slice(s, s + micro_train_batch_size)
                rm = response_mask[sl]
                if rm.sum() == 0:
                    continue

                out = get_response_log_probs(
                    policy, input_ids[sl], labels[sl], return_token_entropy=True
                )
                examples_seen += int(input_ids[sl].shape[0])
                tokens_seen += int(input_ids[sl].numel())
                log_probs = out["log_probs"]
                token_entropy = out["token_entropy"]
                denom = rm.sum().to(token_entropy.dtype).clamp(min=1)
                avg_ent = (token_entropy * rm).sum() / denom

                old_slice = old_full[sl] if old_full is not None else None
                mb_loss, meta = grpo_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=rm,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_literal,
                    raw_rewards=raw_b[sl] if loss_type == "no_baseline" else None,
                    advantages=adv_b[sl] if loss_type != "no_baseline" else None,
                    old_log_probs=old_slice,
                    cliprange=cliprange if loss_type == "grpo_clip" else None,  # grpo_no_clip ignores cliprange
                    length_norm=length_norm,
                )

                mb_total += 1
                accum_loss += float(mb_loss.item())
                accum_ent += float(avg_ent.item())
                if "clip_fraction" in meta:
                    accum_clip += float(meta["clip_fraction"].item())
                n_accum_logs += 1

                if mb_total % gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    log_payload = {
                        "train/loss": accum_loss / max(n_accum_logs, 1),
                        "train/grad_norm": float(grad_norm),
                        "train/token_entropy": accum_ent / max(n_accum_logs, 1),
                        "grpo_step": grpo_step,
                        "examples_seen": examples_seen,
                        "tokens_seen": tokens_seen,
                    }
                    if loss_type == "grpo_clip":
                        log_payload["train/clip_fraction"] = accum_clip / max(n_accum_logs, 1)
                    wandb.log(log_payload)
                    accum_loss = 0.0
                    accum_clip = 0.0
                    accum_ent = 0.0
                    n_accum_logs = 0

        if mb_total % gradient_accumulation_steps != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            log_payload = {
                "train/loss": accum_loss / max(n_accum_logs, 1),
                "train/grad_norm": float(grad_norm),
                "train/token_entropy": accum_ent / max(n_accum_logs, 1),
                "grpo_step": grpo_step,
                "examples_seen": examples_seen,
                "tokens_seen": tokens_seen,
            }
            if loss_type == "grpo_clip":
                log_payload["train/clip_fraction"] = accum_clip / max(n_accum_logs, 1)
            wandb.log(log_payload)

        load_policy_into_vllm(policy, llm)
        typer.echo(f"  Train reward (mean): {reward_stats['train/reward_mean']:.4f}")

        if grpo_step % val_every == 0 or grpo_step == n_grpo_steps:
            eval_prompts = val_prompts
            eval_answers = val_answers
            if grpo_step == n_grpo_steps:
                typer.echo(f"  Final validation on {len(final_val_answers)} examples ...")
                eval_prompts = final_val_prompts
                eval_answers = final_val_answers
                eval_output_path = final_eval_output_path
            else:
                eval_output_path = None
            val_out = run_validation(
                policy, llm, eval_prompts, eval_answers, tokenizer, eval_sampling, grpo_step,
                examples_seen, tokens_seen, reward_fn, eval_output_path,
            )
            r = float(val_out["metrics"]["avg_reward"])
            typer.echo(f"  eval avg_reward={r:.4f}  acc={val_out['accuracy']:.4f}")
            if eval_output_path is not None:
                typer.echo(f"  Final eval results saved to {final_eval_output_path}")

            samples = []
            for rec in val_out["records"][:3]:
                samples.append({
                    "grpo_step": grpo_step,
                    "prompt": rec["prompt"][:400],
                    "response": rec["response"][:800],
                    "ground_truth": rec["ground_truth"],
                    "reward": rec["reward"],
                })
            with open(rollout_log_path, "a") as f:
                f.write(json.dumps({"grpo_step": grpo_step, "samples": samples}) + "\n")

        if grpo_step % 50 == 0:
            ckpt_dir = out_dir / f"checkpoint_step_{grpo_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            typer.echo(f"  Step-{grpo_step} checkpoint saved to {ckpt_dir}")

    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    typer.echo(f"\nCheckpoint saved to {ckpt_dir}")
    typer.echo(f"Rollout log: {rollout_log_path}")
    wandb.finish()


if __name__ == "__main__":
    app()
