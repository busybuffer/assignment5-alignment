"""
DPO fine-tuning of an instruction-tuned Llama 3.1 8B model on Anthropic HH.

Uses two GPUs: one for the reference model (frozen) and one for the policy model
being trained. Gradient accumulation is used to achieve larger effective batch sizes.
Optimizer: RMSprop (as in the original DPO paper).

Usage:
    python scripts/train_dpo.py \
        --model-id outputs/sft/llama-3.1-8b-sft \
        --hh-data-dir data/hh \
        --output-dir outputs/dpo/llama-3.1-8b-dpo \
        --run-name llama-3.1-8b-dpo

    # Quick smoke-test:
    python scripts/train_dpo.py \
        --model-id outputs/sft/llama-3.1-8b-sft \
        --hh-data-dir data/hh \
        --output-dir outputs/dpo/smoke \
        --run-name dpo-smoke \
        --max-steps 10 --grad-accum-steps 1 \
        --no-wandb
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.data import ALPACA_TEMPLATE, load_hh_dataset
from cs336_alignment.dpo import _batch_sequence_log_probs, compute_per_instance_dpo_loss

app = typer.Typer()




@app.command()
def main(
    # Model
    model_id: str = typer.Option(
        ..., "--model-id",
        help="HuggingFace model ID or local path to the SFT model.",
    ),
    # Data
    hh_data_dir: Path = typer.Option(
        Path("data/hh"), "--hh-data-dir",
        help="Directory containing Anthropic HH subsets.",
    ),
    val_size: int = typer.Option(
        200, "--val-size",
        help="Number of examples held out for validation.",
    ),
    # Output
    output_dir: Path = typer.Option(
        Path("outputs/dpo/model"), "--output-dir",
        help="Directory to save the best model.",
    ),
    run_name: str = typer.Option("dpo", "--run-name"),
    # Training hyperparameters
    beta: float = typer.Option(0.1, "--beta", help="DPO beta hyperparameter."),
    grad_accum_steps: int = typer.Option(64, "--grad-accum-steps"),
    max_steps: int = typer.Option(-1, "--max-steps", help="-1 = full epoch."),
    lr: float = typer.Option(1e-6, "--lr"),
    # Logging
    log_interval: int = typer.Option(10, "--log-interval"),
    val_interval: int = typer.Option(100, "--val-interval"),
    val_batches: int = typer.Option(200, "--val-batches",
                                    help="Number of validation examples to evaluate."),
    # Devices
    policy_device: str = typer.Option("cuda:0", "--policy-device"),
    ref_device: str = typer.Option("cuda:1", "--ref-device"),
    dtype: str = typer.Option("bfloat16", "--dtype"),
    seed: int = typer.Option(42, "--seed"),
    resume_from: str = typer.Option(
        "",
        "--resume-from",
        help="Path to a checkpoint to resume from (e.g. outputs/dpo/model/best). "
             "Must also pass --resume-step so the trainer knows how many examples to skip.",
    ),
    resume_step: int = typer.Option(
        0,
        "--resume-step",
        help="Optimizer step at which the checkpoint was saved. "
             "Used to skip already-processed training examples.",
    ),
    # W&B
    wandb_project: str = typer.Option("cs336-dpo", "--wandb-project"),
    use_wandb: bool = typer.Option(True, "--wandb/--no-wandb"),
):
    random.seed(seed)
    torch.manual_seed(seed)

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    # ------------------------------------------------------------------ wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "model_id": model_id,
                "beta": beta,
                "grad_accum_steps": grad_accum_steps,
                "lr": lr,
            },
        )

    # ------------------------------------------------------------------ data
    typer.echo("Loading HH dataset ...")
    all_examples = load_hh_dataset(hh_data_dir, split="train")
    random.shuffle(all_examples)

    val_examples = all_examples[:val_size]
    train_examples = all_examples[val_size:]
    typer.echo(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    # ------------------------------------------------------------------ models
    policy_ckpt = resume_from if resume_from else model_id
    typer.echo(f"Loading policy model from {policy_ckpt} on {policy_device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(
        policy_ckpt,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    ).to(policy_device)
    policy.train()

    typer.echo(f"Loading reference model from {model_id} on {ref_device} ...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    ).to(ref_device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------ optimizer
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=lr)

    # ------------------------------------------------------------------ val helper
    eos = tokenizer.eos_token or ""

    def tok(text: str, device: str) -> torch.Tensor:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.long, device=device)

    def run_validation() -> float:
        policy.eval()
        correct = 0
        for ex in val_examples[:val_batches]:
            chosen_text = ALPACA_TEMPLATE.format(
                instruction=ex["instruction"], response=ex["chosen"]
            ) + eos
            rejected_text = ALPACA_TEMPLATE.format(
                instruction=ex["instruction"], response=ex["rejected"]
            ) + eos

            # Batch chosen + rejected in one forward pass
            log_probs = _batch_sequence_log_probs(
                policy,
                [tok(chosen_text, policy_device), tok(rejected_text, policy_device)],
                no_grad=True,
            )
            correct += int(log_probs[0].item() > log_probs[1].item())

        policy.train()
        return correct / len(val_examples[:val_batches])

    # ------------------------------------------------------------------ training
    best_val_acc = 0.0
    optimizer_step = resume_step
    microbatch_idx = 0
    accumulated_loss = 0.0

    # Skip examples already processed before the resume checkpoint
    skip_examples = resume_step * grad_accum_steps
    if skip_examples > 0:
        typer.echo(f"Resuming from step {resume_step}, skipping {skip_examples} examples ...")
        train_examples = train_examples[skip_examples:]

    total_steps = (len(train_examples) // grad_accum_steps) if max_steps < 0 else max_steps
    typer.echo(f"Remaining optimizer steps: {total_steps}")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_dir = output_dir / "best"

    typer.echo("Starting DPO training ...")
    for ex in train_examples:
        if max_steps > 0 and optimizer_step >= max_steps:
            break

        # Compute DPO loss for this example
        loss = compute_per_instance_dpo_loss(
            lm=policy,
            lm_ref=ref_model,
            tokenizer=tokenizer,
            beta=beta,
            prompt=ex["instruction"],
            response_chosen=ex["chosen"],
            response_rejected=ex["rejected"],
        )

        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()
        accumulated_loss += loss.item()
        microbatch_idx += 1

        if microbatch_idx % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step += 1

            avg_loss = accumulated_loss / grad_accum_steps
            accumulated_loss = 0.0

            if optimizer_step % log_interval == 0:
                typer.echo(f"  step={optimizer_step}  loss={avg_loss:.4f}")

            if use_wandb:
                import wandb
                wandb.log({"train/loss": avg_loss, "optimizer_step": optimizer_step})

            if optimizer_step % val_interval == 0:
                val_acc = run_validation()
                typer.echo(f"  => val_acc={val_acc:.4f} at step {optimizer_step}")

                if use_wandb:
                    import wandb
                    wandb.log({"val/accuracy": val_acc, "optimizer_step": optimizer_step})

                # Save checkpoint for every val step
                ckpt_dir = output_dir / f"step-{optimizer_step}"
                policy.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                typer.echo(f"  Saved checkpoint to {ckpt_dir}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    typer.echo(f"  New best val_acc={best_val_acc:.4f}, saving best ...")
                    policy.save_pretrained(best_ckpt_dir)
                    tokenizer.save_pretrained(best_ckpt_dir)

    # Flush remaining gradients
    if microbatch_idx % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        optimizer_step += 1

    # Final validation
    val_acc = run_validation()
    typer.echo(f"\nFinal val_acc: {val_acc:.4f}  |  Best val_acc: {best_val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        policy.save_pretrained(best_ckpt_dir)
        tokenizer.save_pretrained(best_ckpt_dir)

    # Also save final model
    final_dir = output_dir / "final"
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    typer.echo(f"Best model saved to {best_ckpt_dir}")
    typer.echo(f"Final model saved to {final_dir}")

    if use_wandb:
        import wandb
        wandb.log({"val/accuracy_final": val_acc, "val/accuracy_best": best_val_acc})
        wandb.finish()


if __name__ == "__main__":
    app()
