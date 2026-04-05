from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    # Tokenize prompts and outputs separately (no special tokens on outputs to avoid double BOS)
    prompt_ids = [tokenizer.encode(p, add_special_tokens=True) for p in prompt_strs]
    output_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    combined = [p + o for p, o in zip(prompt_ids, output_ids)]
    max_len = max(len(seq) for seq in combined)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Pad all sequences to max_len
    full_input_ids = torch.full((len(combined), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(combined):
        full_input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    input_ids = full_input_ids[:, :-1]  # drop last token
    labels = full_input_ids[:, 1:]      # drop first token (shifted labels)

    # response_mask: 1 for positions in `labels` that correspond to output tokens
    # labels[i, j] == full_input_ids[i, j+1], so output starts at index len(prompt_ids[i])-1
    response_mask = torch.zeros_like(labels)
    for i, (p_ids, seq) in enumerate(zip(prompt_ids, combined)):
        resp_start = len(p_ids) - 1      # first output token lands here in labels
        resp_end = len(seq) - 1          # last output token (exclusive)
        response_mask[i, resp_start:resp_end] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
