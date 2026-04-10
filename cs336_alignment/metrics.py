from __future__ import annotations

import re
from typing import Any


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Parse a model output into a predicted option letter (A, B, C, or D).
    Returns None if the output cannot be parsed.
    """
    # Look for "The correct answer is X" pattern
    match = re.search(r"[Tt]he correct answer is\s+([A-D])\b", model_output)
    if match:
        return match.group(1).upper()

    # Fallback: look for a standalone letter A/B/C/D
    match = re.search(r"\b([A-D])\b", model_output)
    if match:
        return match.group(1).upper()

    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    """
    Parse a model output into a numeric prediction by taking the last number
    in the output. Returns None if no number is found.
    """
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", model_output)
    if not matches:
        return None
    # Strip commas from numbers like "1,000"
    return matches[-1].replace(",", "")
