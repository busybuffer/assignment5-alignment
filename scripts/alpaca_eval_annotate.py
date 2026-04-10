"""
Compute AlpacaEval winrate using Llama 3.3 70B via a free API.

Compares our model's outputs against the official GPT-4 Turbo reference
outputs from the AlpacaEval HuggingFace dataset (tatsu-lab/alpaca_eval),
using Llama 3.3 70B Instruct as the annotator.

Each pair is judged twice (swapping A/B order) to control for position bias.
Winrate and length-controlled winrate are computed and saved.

Supported backends (both free):
  groq      -- https://console.groq.com      (GROQ_API_KEY)      100k TPD limit
  cerebras  -- https://cloud.cerebras.ai     (CEREBRAS_API_KEY)  faster, higher limits

Setup:
    pip install groq openai scikit-learn  # openai SDK used for cerebras

Usage:
    # Groq backend (default):
    export GROQ_API_KEY="gsk_..."
    python scripts/alpaca_eval_annotate.py \
        --model-outputs eval/alpaca_eval_baseline.json \
        --output-path outputs/alpaca_eval_annotated.jsonl

    # Cerebras backend:
    export CEREBRAS_API_KEY="csk_..."
    python scripts/alpaca_eval_annotate.py \
        --model-outputs eval/alpaca_eval_baseline.json \
        --output-path outputs/alpaca_eval_annotated.jsonl \
        --backend cerebras

    # Resume interrupted run:
    python scripts/alpaca_eval_annotate.py \
        --model-outputs eval/alpaca_eval_baseline.json \
        --output-path outputs/alpaca_eval_annotated.jsonl \
        --resume
    
    python3 scripts/alpaca_eval_annotate.py \
        --model-outputs eval/alpaca_eval_baseline.json \
        --output-path outputs/alpaca_eval_annotated.jsonl \
        --backend cerebras \
        --resume   # picks up where Groq left off
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import typer

app = typer.Typer()

# Backend configs: model name, RPM limit, env var for API key, base URL (None = SDK default)
BACKEND_CONFIGS = {
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "rpm": 28,           # free tier: 30 RPM
        "env_key": "GROQ_API_KEY",
        "base_url": None,    # use groq SDK
    },
    "cerebras": {
        "model": "llama-3.3-70b-instruct",
        "rpm": 28,
        "env_key": "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
    },
    "openrouter": {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "rpm": 18,           # free tier: 20 RPM
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "together": {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "rpm": 58,           # free tier: 60 RPM
        "env_key": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
    },
}

# Official GPT-4 Turbo (gpt4_1106_preview) reference outputs.
# Download with:
#   wget "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval_gpt4_baseline.json"
GPT4_TURBO_DEFAULT_PATH = Path("data/alpaca_eval/alpaca_eval_gpt4_baseline.json")

ANNOTATOR_PROMPT = """\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. \
You should choose the assistant that follows the user's instructions and answers the user's question better. \
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. \
Begin your evaluation by comparing the two responses and provide a short explanation. \
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. \
Do not allow the length of the responses to influence your evaluation. \
Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \
"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{instruction}

[The Start of Assistant A's Answer]
{output_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{output_b}
[The End of Assistant B's Answer]"""


def load_gpt4_turbo_reference(path: Path) -> dict[str, str]:
    """
    Load official GPT-4 Turbo reference outputs from local JSON file.
    Download with:
      wget "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval_gpt4_baseline.json"
    Returns a dict mapping instruction -> gpt4_turbo_output.
    """
    typer.echo(f"Loading GPT-4 Turbo reference outputs from {path} ...")
    with open(path) as f:
        data = json.load(f)
    ref = {row["instruction"]: row["output"] for row in data}
    typer.echo(f"Loaded {len(ref)} GPT-4 Turbo reference outputs.")
    return ref


def parse_preference(text: str) -> str | None:
    """Return 'A' or 'B' from annotator response, or None if unparseable."""
    match = re.search(r"\[\[([AB])\]\]", text)
    if match:
        return match.group(1)
    matches = re.findall(r"\b([AB])\b", text)
    return matches[-1] if matches else None


def make_client(backend: str, api_key: str):
    """Return an OpenAI-compatible client for the given backend."""
    cfg = BACKEND_CONFIGS[backend]
    if backend == "groq":
        from groq import Groq
        return Groq(api_key=api_key)
    else:
        # Cerebras and others expose an OpenAI-compatible API
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=cfg["base_url"])


def call_api(client, model: str, instruction: str, output_a: str, output_b: str,
             max_retries: int = 5) -> str | None:
    """Call the annotator API and return 'A', 'B', or None. Retries on 429."""
    prompt = ANNOTATOR_PROMPT.format(
        instruction=instruction,
        output_a=output_a,
        output_b=output_b,
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            return parse_preference(response.choices[0].message.content)
        except Exception as e:
            msg = str(e)
            if "429" in msg and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)  # 30s, 60s, 90s, 120s
                typer.echo(f"  Rate limited, waiting {wait}s before retry {attempt+1}/{max_retries-1}...", err=True)
                time.sleep(wait)
            else:
                typer.echo(f"  API error: {e}", err=True)
                return None
    return None


def compute_winrate(records: list[dict]) -> dict[str, float]:
    """
    Compute winrates and length-controlled winrate.

    Returns a dict with:
      winrate_a:      winrate when ours=A, ref=B  ('A' = ours wins)
      winrate_b:      winrate when ref=A, ours=B  ('B' = ours wins)
      winrate:        debiased winrate averaged over both orderings
      lc_winrate:     length-controlled winrate (logistic regression at equal lengths)
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    wins_a, wins_b = [], []   # per-ordering win indicators
    wins = []
    len_ratios = []

    for r in records:
        pref_ours_first = r.get("pref_ours_first")  # 'A' = ours wins
        pref_ref_first = r.get("pref_ref_first")    # 'B' = ours wins

        if pref_ours_first is not None:
            wins_a.append(1.0 if pref_ours_first == "A" else 0.0)
        if pref_ref_first is not None:
            wins_b.append(1.0 if pref_ref_first == "B" else 0.0)

        votes = wins_a[-1:] + wins_b[-1:]  # whichever were just appended
        votes = []
        if pref_ours_first is not None:
            votes.append(1.0 if pref_ours_first == "A" else 0.0)
        if pref_ref_first is not None:
            votes.append(1.0 if pref_ref_first == "B" else 0.0)
        if not votes:
            continue

        wins.append(sum(votes) / len(votes))
        our_len = len(r["our_output"])
        ref_len = len(r["ref_output"])
        len_ratios.append(math.log(max(our_len, 1) / max(ref_len, 1)))

    winrate_a = sum(wins_a) / len(wins_a) if wins_a else 0.0
    winrate_b = sum(wins_b) / len(wins_b) if wins_b else 0.0
    winrate = sum(wins) / len(wins) if wins else 0.0

    X = np.array([[w, lr] for w, lr in zip(wins, len_ratios)])
    y = np.array(wins)
    try:
        clf = LogisticRegression(fit_intercept=True, max_iter=1000)
        clf.fit(X, y)
        X_neutral = np.array([[w, 0.0] for w in wins])
        lc_winrate = float(clf.predict_proba(X_neutral)[:, 1].mean())
    except Exception:
        lc_winrate = winrate

    return {
        "winrate_ours_as_A": winrate_a,
        "winrate_ours_as_B": winrate_b,
        "winrate": winrate,
        "lc_winrate": lc_winrate,
    }


@app.command()
def main(
    model_outputs: Path = typer.Option(
        ...,
        "--model-outputs",
        help="Path to our model predictions JSON (alpaca_eval format).",
    ),
    output_path: Path = typer.Option(
        Path("outputs/alpaca_eval_annotated.jsonl"),
        "--output-path",
        help="Path to write per-example annotation results as JSONL.",
    ),
    ref_outputs_path: Path = typer.Option(
        GPT4_TURBO_DEFAULT_PATH,
        "--ref-outputs-path",
        help="Path to alpaca_eval_gpt4_baseline.json (GPT-4 Turbo reference outputs).",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from existing output file, skipping already-annotated examples.",
    ),
    max_examples: int = typer.Option(
        -1,
        "--max-examples",
        help="Maximum number of examples to annotate (-1 for all).",
    ),
    api_key: str = typer.Option(
        "",
        "--api-key",
        help="API key (defaults to the env var for the selected backend).",
    ),
    backend: str = typer.Option(
        "groq",
        "--backend",
        help="Backend to use for annotation: 'groq' or 'cerebras'.",
    ),
):
    if backend not in BACKEND_CONFIGS:
        typer.echo(f"Unknown backend '{backend}'. Choose from: {list(BACKEND_CONFIGS)}")
        raise typer.Exit(1)

    cfg = BACKEND_CONFIGS[backend]
    model = cfg["model"]
    rpm = cfg["rpm"]
    seconds_per_request = 60.0 / rpm

    key = api_key or os.environ.get(cfg["env_key"], "")
    if not key:
        typer.echo(f"Set {cfg['env_key']} env var or pass --api-key")
        raise typer.Exit(1)

    client = make_client(backend, key)
    typer.echo(f"Using backend: {backend} | model: {model}")

    # Load our model outputs
    with open(model_outputs) as f:
        our_data = json.load(f)
    our_outputs = {ex["instruction"]: ex["output"] for ex in our_data}
    typer.echo(f"Loaded {len(our_outputs)} model outputs from {model_outputs}.")

    # Load GPT-4 Turbo reference outputs from local file
    ref_outputs = load_gpt4_turbo_reference(ref_outputs_path)

    instructions = list(our_outputs.keys())
    if max_examples > 0:
        instructions = instructions[:max_examples]

    # Check coverage
    missing = [ins for ins in instructions if ins not in ref_outputs]
    if missing:
        typer.echo(f"Warning: {len(missing)} instructions have no GPT-4 Turbo reference output.")

    # Resume from existing annotations
    done: dict[str, dict] = {}
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["instruction"]] = r
        typer.echo(f"Resuming: {len(done)} already annotated.")

    remaining = [ins for ins in instructions if ins not in done and ins in ref_outputs]
    typer.echo(f"To annotate: {len(remaining)} instructions (2 API calls each).")
    typer.echo(f"Estimated time at {rpm} RPM: ~{len(remaining) * 2 / rpm:.1f} minutes")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a") as out_f:
        for i, instruction in enumerate(remaining):
            our_out = our_outputs[instruction]
            ref_out = ref_outputs[instruction]

            typer.echo(f"[{i+1}/{len(remaining)}] Annotating...")

            # Order 1: ours=A, gpt4-turbo=B
            t0 = time.perf_counter()
            pref_ours_first = call_api(client, model, instruction, our_out, ref_out)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, seconds_per_request - elapsed))

            # Order 2: gpt4-turbo=A, ours=B
            t0 = time.perf_counter()
            pref_ref_first = call_api(client, model, instruction, ref_out, our_out)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, seconds_per_request - elapsed))

            record = {
                "instruction": instruction,
                "our_output": our_out,
                "ref_output": ref_out,
                "pref_ours_first": pref_ours_first,  # 'A'=ours wins
                "pref_ref_first": pref_ref_first,    # 'B'=ours wins
            }
            done[instruction] = record
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    all_records = list(done.values())
    metrics = compute_winrate(all_records)

    typer.echo(f"\n=== AlpacaEval Results ===")
    typer.echo(f"Reference model:        GPT-4 Turbo (gpt4_1106_preview)")
    typer.echo(f"Annotator:              {model} via {backend}")
    typer.echo(f"Examples annotated:     {len(all_records)}")
    typer.echo(f"Winrate (ours=A):       {metrics['winrate_ours_as_A']:.4f} ({metrics['winrate_ours_as_A']*100:.2f}%)")
    typer.echo(f"Winrate (ours=B):       {metrics['winrate_ours_as_B']:.4f} ({metrics['winrate_ours_as_B']*100:.2f}%)")
    typer.echo(f"Winrate (debiased):     {metrics['winrate']:.4f} ({metrics['winrate']*100:.2f}%)")
    typer.echo(f"Length-controlled WR:   {metrics['lc_winrate']:.4f} ({metrics['lc_winrate']*100:.2f}%)")
    typer.echo(f"\nResults saved to {output_path}")

    summary = {
        "reference_model": "gpt4_turbo (gpt4_1106_preview)",
        "annotator": model,
        "backend": backend,
        "n_examples": len(all_records),
        "winrate_ours_as_A": round(metrics["winrate_ours_as_A"], 4),
        "winrate_ours_as_B": round(metrics["winrate_ours_as_B"], 4),
        "winrate": round(metrics["winrate"], 4),
        "length_controlled_winrate": round(metrics["lc_winrate"], 4),
    }
    typer.echo("\nSummary:")
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
