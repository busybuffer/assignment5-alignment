from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
)


class PackedSFTDataset(Dataset):
    """
    PyTorch Dataset for packed instruction fine-tuning.

    Concatenates (prompt, response) pairs formatted with the Alpaca template,
    separated by the tokenizer's EOS token, then splits the resulting token
    sequence into non-overlapping chunks of `seq_length`. The final incomplete
    chunk is dropped.

    __getitem__ returns:
        input_ids: LongTensor of shape (seq_length,)  — tokens [i*L : i*L+L]
        labels:    LongTensor of shape (seq_length,)  — tokens [i*L+1 : i*L+L+1]
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool,
    ) -> None:
        self.seq_length = seq_length

        # Load examples
        examples = []
        with open(dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        if shuffle:
            random.shuffle(examples)

        # Format each example with the Alpaca template
        eos = tokenizer.eos_token_id

        all_ids: list[int] = []
        for ex in examples:
            text = ALPACA_TEMPLATE.format(
                instruction=ex["prompt"],
                response=ex["response"],
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(eos)  # delimiter between documents

        # all_ids is the full token sequence; we need seq_length+1 tokens per
        # chunk so we can produce both input_ids and labels (shifted by 1).
        # We split into non-overlapping windows of (seq_length + 1) and drop
        # the last incomplete window.
        chunk = seq_length + 1
        n_chunks = len(all_ids) // chunk
        trimmed = all_ids[: n_chunks * chunk]

        # Shape: (n_chunks, seq_length+1)
        token_matrix = torch.tensor(trimmed, dtype=torch.long).view(n_chunks, chunk)

        # input_ids: first seq_length tokens of each chunk
        # labels:    last seq_length tokens of each chunk (shifted by 1)
        self._input_ids = token_matrix[:, :-1]  # (n_chunks, seq_length)
        self._labels = token_matrix[:, 1:]       # (n_chunks, seq_length)

    def __len__(self) -> int:
        return self._input_ids.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._input_ids[i],
            "labels": self._labels[i],
        }


def _parse_hh_conversation(text: str) -> tuple[str, str] | None:
    """
    Parse an HH conversation string into (instruction, response).

    Returns None for multi-turn conversations (human sent >1 message)
    or malformed text.

    Format: "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ..."
    We accept only single-turn: exactly one Human turn followed by one Assistant turn.
    """
    roles = re.findall(r"\n\n(Human|Assistant): ", text)
    if len(roles) != 2 or roles[0] != "Human" or roles[1] != "Assistant":
        return None  # multi-turn or malformed

    # Extract content of each turn
    parts = re.split(r"\n\nHuman: |\n\nAssistant: ", text)
    parts = [p for p in parts if p]  # drop leading empty string

    if len(parts) < 2:
        return None

    return parts[0].strip(), parts[1].strip()


# Subset names to load (red-team-attempts excluded — no chosen/rejected pairs)
HH_SUBSETS = [
    "helpful-base",
    "helpful-online",
    "helpful-rejection-sampled",
    "harmless-base",
]


def load_hh_dataset(data_dir: str | Path, split: str = "train") -> list[dict]:
    """
    Load the Anthropic HH-RLHF dataset.

    Reads all four subsets (helpful-base, helpful-online,
    helpful-rejection-sampled, harmless-base) for the given split
    and returns a combined list of examples.

    Each returned example is a dict with:
        instruction (str): the first human message
        chosen      (str): the preferred assistant response
        rejected    (str): the dispreferred assistant response
        source      (str): which subset the example came from

    Multi-turn conversations (human sent >1 message) are dropped.

    Args:
        data_dir: path to the directory containing the HH subsets
                  (e.g. "data/hh")
        split:    "train" or "test"
    """
    data_dir = Path(data_dir)
    examples = []

    for subset in HH_SUBSETS:
        path = data_dir / subset / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                chosen_parsed = _parse_hh_conversation(obj["chosen"])
                rejected_parsed = _parse_hh_conversation(obj["rejected"])

                # Drop multi-turn or mismatched examples
                if chosen_parsed is None or rejected_parsed is None:
                    continue
                if chosen_parsed[0] != rejected_parsed[0]:
                    # Instructions must match — they should share the same prompt
                    continue

                instruction, chosen_response = chosen_parsed
                _, rejected_response = rejected_parsed

                examples.append({
                    "instruction": instruction,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                    "source": subset,
                })

    return examples


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Return a DataLoader that iterates over the dataset in batches."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
