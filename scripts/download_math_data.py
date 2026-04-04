"""
Download qwedsacf/competition_math from HuggingFace and save as JSONL.

Saves:
    train.jsonl  — 7500 examples
    valid.jsonl  — 5000 examples

Usage:
    uv run python scripts/download_math_data.py --output-dir data/MATH
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import typer
from huggingface_hub import snapshot_download

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

    typer.echo("Downloading qwedsacf/competition_math parquet files ...")
    local_dir = snapshot_download(
        repo_id="qwedsacf/competition_math",
        repo_type="dataset",
        ignore_patterns=["*.md", "*.gitattributes"],
    )

    parquet_files = sorted(Path(local_dir).glob("**/*.parquet"))
    typer.echo(f"Found {len(parquet_files)} parquet file(s)")

    table = pq.ParquetDataset([str(f) for f in parquet_files]).read()
    rows = table.to_pydict()
    total = len(rows["problem"])
    typer.echo(f"Total examples: {total}")

    splits = [
        ("train", range(0, TRAIN_SIZE)),
        ("valid", range(TRAIN_SIZE, TRAIN_SIZE + VALID_SIZE)),
    ]
    for split_name, idx_range in splits:
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for i in idx_range:
                answer = extract_boxed_answer(rows["solution"][i])
                record = {
                    "problem": rows["problem"][i],
                    "solution": rows["solution"][i],
                    "answer": answer,
                    "level": rows["level"][i],
                    "type": rows["type"][i],
                }
                f.write(json.dumps(record) + "\n")
        typer.echo(f"Saved {len(list(idx_range))} examples to {out_path}")


if __name__ == "__main__":
    app()
