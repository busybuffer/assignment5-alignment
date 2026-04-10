from __future__ import annotations

import json
import random
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


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Return a DataLoader that iterates over the dataset in batches."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
