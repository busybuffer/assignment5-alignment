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

# Maps output filename → HuggingFace split name
SPLIT_MAP = {
    "train": "train",   # 7500 examples
    "valid": "test",    # 5000 examples
}


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("data/math"),
        "--output-dir",
        help="Directory to save the JSONL files.",
    ),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for out_name, hf_split in SPLIT_MAP.items():
        typer.echo(f"Downloading qwedsacf/competition_math / {hf_split} -> {out_name}.jsonl ...")
        ds = load_dataset("qwedsacf/competition_math", split=hf_split)

        out_path = output_dir / f"{out_name}.jsonl"
        with open(out_path, "w") as f:
            for example in ds:
                answer = extract_boxed_answer(example["solution"])
                record = {
                    "problem": example["problem"],
                    "solution": example["solution"],
                    "answer": answer,
                    "level": example["level"],
                    "type": example["type"],
                }
                f.write(json.dumps(record) + "\n")

        typer.echo(f"Saved {len(ds)} examples to {out_path}")


if __name__ == "__main__":
    app()
