"""
Instruction fine-tuning of Llama 3.1 8B on the safety-augmented UltraChat dataset.

Uses packed sequences, gradient accumulation, and periodic validation loss logging.
Optionally logs to Weights & Biases.

Usage:
    python scripts/train_instruction_tuning.py \
        --model-id meta-llama/Meta-Llama-3.1-8B \
        --train-data data/sft/train.jsonl \
        --val-data data/sft/test.jsonl \
        --output-dir outputs/sft/llama-3.1-8b-sft \
        --run-name llama-3.1-8b-sft

    # Quick smoke-test (small batch, few steps):
    python scripts/train_instruction_tuning.py \
        --model-id meta-llama/Meta-Llama-3.1-8B \
        --train-data data/sft/train.jsonl \
        --val-data data/sft/test.jsonl \
        --output-dir outputs/sft/smoke \
        --run-name smoke \
        --max-steps 10 --batch-size 2 --grad-accum-steps 1 \
        --no-wandb
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.data import PackedSFTDataset

app = typer.Typer()


@app.command()
def main(
    # Model
    model_id: str = typer.Option(
        ..., "--model-id",
        help="HuggingFace model ID or local path to Llama 3.1 8B.",
    ),
    # Data
    train_data: Path = typer.Option(
        Path("data/sft/train.jsonl"), "--train-data",
        help="Path to training JSONL.",
    ),
    val_data: Path = typer.Option(
        Path("data/sft/test.jsonl"), "--val-data",
        help="Path to validation JSONL.",
    ),
    # Output
    output_dir: Path = typer.Option(
        Path("outputs/sft/model"), "--output-dir",
        help="Directory to save the final model.",
    ),
    run_name: str = typer.Option("sft", "--run-name"),
    # Training hyperparameters
    seq_length: int = typer.Option(512, "--seq-length", help="Packed sequence length."),
    batch_size: int = typer.Option(2, "--batch-size", help="Microbatch size (sequences per step)."),
    grad_accum_steps: int = typer.Option(16, "--grad-accum-steps", help="Gradient accumulation steps. Effective batch = batch_size * grad_accum_steps."),
    num_epochs: int = typer.Option(1, "--num-epochs"),
    max_steps: int = typer.Option(-1, "--max-steps", help="Stop after this many optimizer steps (-1 = full training)."),
    lr: float = typer.Option(2e-5, "--lr"),
    weight_decay: float = typer.Option(0.0, "--weight-decay"),
    max_grad_norm: float = typer.Option(1.0, "--max-grad-norm"),
    warmup_fraction: float = typer.Option(0.03, "--warmup-fraction", help="Fraction of total steps used for linear warmup."),
    # Logging
    log_interval: int = typer.Option(10, "--log-interval", help="Log every N optimizer steps."),
    val_interval: int = typer.Option(200, "--val-interval", help="Run validation every N optimizer steps."),
    val_batches: int = typer.Option(50, "--val-batches", help="Number of validation batches to average over."),
    # Infrastructure
    device: str = typer.Option("cuda:0", "--device"),
    dtype: str = typer.Option("bfloat16", "--dtype", help="'bfloat16' or 'float32'."),
    seed: int = typer.Option(42, "--seed"),
    shuffle_train: bool = typer.Option(True, "--shuffle-train/--no-shuffle-train"),
    # W&B
    wandb_project: str = typer.Option("cs336-sft", "--wandb-project"),
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
                "seq_length": seq_length,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": batch_size * grad_accum_steps,
                "num_epochs": num_epochs,
                "lr": lr,
                "warmup_fraction": warmup_fraction,
            },
        )

    # ------------------------------------------------------------------ data
    typer.echo("Building datasets ...")
    train_dataset = PackedSFTDataset(
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        dataset_path=train_data,
        seq_length=seq_length,
        shuffle=shuffle_train,
    )
    val_dataset = PackedSFTDataset(
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        dataset_path=val_data,
        seq_length=seq_length,
        shuffle=False,
    )
    typer.echo(f"Train: {len(train_dataset)} sequences | Val: {len(val_dataset)} sequences")
    typer.echo(f"Effective batch size: {batch_size * grad_accum_steps} sequences")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # ------------------------------------------------------------------ model
    typer.echo(f"Loading model {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device)
    model.train()

    # ------------------------------------------------------------------ optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Compute total optimizer steps to set up cosine decay schedule
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = num_epochs * steps_per_epoch if max_steps < 0 else max_steps
    warmup_steps = max(1, int(warmup_fraction * total_steps))
    typer.echo(f"Total optimizer steps: {total_steps} | Warmup steps: {warmup_steps}")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------ validation helper
    @torch.no_grad()
    def run_validation() -> float:
        model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in val_loader:
            if n_batches >= val_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            total_loss += loss.item()
            n_batches += 1
        model.train()
        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------ training loop
    optimizer_step = 0
    microbatch_idx = 0
    accumulated_loss = 0.0
    examples_seen = 0
    tokens_seen = 0

    typer.echo("Starting training ...")
    for epoch in range(num_epochs):
        typer.echo(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        for batch in train_loader:
            if max_steps > 0 and optimizer_step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            accumulated_loss += loss.item()
            examples_seen += input_ids.shape[0]
            tokens_seen += input_ids.numel()
            microbatch_idx += 1

            if microbatch_idx % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                avg_loss = accumulated_loss / grad_accum_steps
                accumulated_loss = 0.0
                current_lr = scheduler.get_last_lr()[0]

                if optimizer_step % log_interval == 0:
                    typer.echo(
                        f"  step={optimizer_step}  loss={avg_loss:.4f}  "
                        f"lr={current_lr:.2e}  "
                        f"examples={examples_seen}  tokens={tokens_seen}"
                    )

                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "train/epoch": epoch + 1,
                        "examples_seen": examples_seen,
                        "tokens_seen": tokens_seen,
                        "optimizer_step": optimizer_step,
                    })

                if optimizer_step % val_interval == 0:
                    val_loss = run_validation()
                    typer.echo(f"  => val_loss={val_loss:.4f} at step {optimizer_step}")
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "val/loss": val_loss,
                            "optimizer_step": optimizer_step,
                        })

        if max_steps > 0 and optimizer_step >= max_steps:
            typer.echo(f"Reached max_steps={max_steps}, stopping early.")
            break

    # Flush any remaining accumulated gradients
    if microbatch_idx % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        optimizer_step += 1

    # ------------------------------------------------------------------ final val
    val_loss = run_validation()
    typer.echo(f"\nFinal val_loss: {val_loss:.4f}")
    if use_wandb:
        import wandb
        wandb.log({"val/loss_final": val_loss, "optimizer_step": optimizer_step})

    # ------------------------------------------------------------------ save
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    typer.echo(f"Model saved to {output_dir}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    app()
