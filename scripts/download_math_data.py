"""
Download the EleutherAI/hendrycks_math dataset from HuggingFace and save as JSONL.

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

SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("data/math"),
        "--output-dir",
        help="Directory to save the JSONL files.",
    ),
    splits: list[str] = typer.Option(
        ["train", "test"],
        "--splits",
        help="Dataset splits to download.",
    ),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        split_records = []
        for subset in SUBSETS:
            typer.echo(f"Downloading {subset} / {split} ...")
            ds = load_dataset(
                "EleutherAI/hendrycks_math",
                subset,
                split=split,
                trust_remote_code=True,
            )
            for example in ds:
                answer = extract_boxed_answer(example["solution"])
                record = {
                    "problem": example["problem"],
                    "solution": example["solution"],
                    "answer": answer,
                    "level": example["level"],
                    "type": example["type"],
                    "subset": subset,
                }
                split_records.append(record)

        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for record in split_records:
                f.write(json.dumps(record) + "\n")
        typer.echo(f"Saved {len(split_records)} examples to {out_path}")


if __name__ == "__main__":
    app()
