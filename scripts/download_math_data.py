"""
Download qwedsacf/competition_math from HuggingFace and save as JSONL.

Saves:
    train.jsonl  — 7500 examples (dataset train split)
    valid.jsonl  — 5000 examples (dataset test split)

Usage:
    uv run python scripts/download_math_data.py --output-dir data/math
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from datasets import load_dataset

from cs336_alignment.drgrpo_grader import extract_boxed_answer

app = typer.Typer()

TRAIN_SIZE = 7500
VALID_SIZE = 5000


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("data/math"),
        "--output-dir",
        help="Directory to save the JSONL files.",
    ),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("Downloading qwedsacf/competition_math (streaming) ...")
    ds = load_dataset("qwedsacf/competition_math", split="train", streaming=True)

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    train_count = valid_count = 0
    with open(train_path, "w") as f_train, open(valid_path, "w") as f_valid:
        for i, example in enumerate(ds):
            answer = extract_boxed_answer(example["solution"])
            record = {
                "problem": example["problem"],
                "solution": example["solution"],
                "answer": answer,
                "level": example["level"],
                "type": example["type"],
            }
            line = json.dumps(record) + "\n"
            if i < TRAIN_SIZE:
                f_train.write(line)
                train_count += 1
            elif i < TRAIN_SIZE + VALID_SIZE:
                f_valid.write(line)
                valid_count += 1
            else:
                break

    typer.echo(f"Saved {train_count} examples to {train_path}")
    typer.echo(f"Saved {valid_count} examples to {valid_path}")


if __name__ == "__main__":
    app()
