from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


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


def _log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-softmax over the last dimension."""
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions over the vocabulary dimension.

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)

    Returns:
        torch.Tensor of shape (batch_size, sequence_length)
    """
    log_probs = _log_softmax(logits)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum masked elements along dim and divide by normalize_constant.

    Args:
        tensor: torch.Tensor to sum and normalize.
        mask: same shape as tensor; 1 = include, 0 = exclude.
        normalize_constant: divisor for normalization.
        dim: dimension to sum along; if None, sum over all dimensions.

    Returns:
        torch.Tensor: normalized sum of masked elements.
    """
    return (tensor * mask).sum(dim=dim) / normalize_constant


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities from a causal language model.

    Args:
        model: HuggingFace causal LM.
        input_ids: shape (batch_size, sequence_length)
        labels: shape (batch_size, sequence_length), shifted input_ids.
        return_token_entropy: if True, also return per-token entropy.

    Returns:
        dict with:
            "log_probs": shape (batch_size, sequence_length)
            "token_entropy": shape (batch_size, sequence_length), only if return_token_entropy=True
    """
    logits = model(input_ids).logits  # (batch, seq_len, vocab)

    # compute log-softmax once and reuse for both log_probs and entropy
    log_probs_all = _log_softmax(logits)
    log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)
    return result
