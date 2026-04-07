from __future__ import annotations

from typing import Callable

import torch
from vllm import LLM, SamplingParams
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


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward-and-backward pass on a single SFT microbatch.

    Args:
        policy_log_probs: (batch_size, sequence_length) per-token log-probs.
        response_mask: (batch_size, sequence_length) 1 for response tokens, 0 otherwise.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        normalize_constant: divisor for the masked sum.

    Returns:
        (loss, metadata): loss is the unscaled scalar for logging;
        backward is called on loss / gradient_accumulation_steps.
    """
    batch_size = policy_log_probs.shape[0]
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant) / (batch_size * gradient_accumulation_steps)
    loss.backward()
    return loss.detach(), {}


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Mean over elements where mask is 1 (or True), same reduction shape as ``tensor.mean(dim)``."""
    m = mask.to(dtype=tensor.dtype)
    return (tensor * m).sum(dim=dim) / m.sum(dim=dim)


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


def log_generations(
    vllm_model: LLM,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params: SamplingParams,
    tokenizer: PreTrainedTokenizerBase,
    label: str = "eval",
) -> dict[str, object]:
    """Generate responses and log per-example and aggregate metrics.

    Args:
        vllm_model: vLLM LLM instance for generation.
        prompts: list of input prompt strings.
        ground_truths: list of ground-truth answer strings (same length as prompts).
        reward_fn: callable(response, ground_truth) -> dict with keys
            "reward", "format_reward", "answer_reward".
        sampling_params: vLLM SamplingParams to use for generation.
        tokenizer: tokenizer used to measure response length in tokens.
        label: short string tag printed in log headers.

    Returns:
        dict with per-example records and aggregate metrics.
    """
    outputs = vllm_model.generate(prompts, sampling_params)
    responses = [out.outputs[0].text for out in outputs]

    records = []
    total_reward = correct_lengths = incorrect_lengths = 0.0
    n_correct = n_incorrect = 0
    all_lengths = []
    all_entropies = []

    for vllm_out, prompt, response, ground_truth in zip(outputs, prompts, responses, ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        response_token_ids = tokenizer.encode(response, add_special_tokens=False)
        resp_len = len(response_token_ids)

        # Approximate per-token entropy from vLLM logprobs if available.
        # vLLM only returns the chosen token's log-prob, so we use H ≈ -log p(chosen)
        # as a proxy (this equals the negative log-likelihood per token).
        token_entropy = None
        lp = vllm_out.outputs[0].logprobs
        if lp:
            token_log_probs = torch.tensor(
                [list(step.values())[0].logprob for step in lp], dtype=torch.float32
            )
            token_entropy = float(-token_log_probs.mean())

        is_correct = reward_dict["answer_reward"] == 1.0
        all_lengths.append(resp_len)
        if is_correct:
            correct_lengths += resp_len
            n_correct += 1
        else:
            incorrect_lengths += resp_len
            n_incorrect += 1
        total_reward += reward_dict["reward"]
        if token_entropy is not None:
            all_entropies.append(token_entropy)

        records.append({
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "reward": reward_dict["reward"],
            "format_reward": reward_dict["format_reward"],
            "answer_reward": reward_dict["answer_reward"],
            "response_length": resp_len,
            "avg_token_entropy": token_entropy,
        })

    n = len(records)
    metrics = {
        "avg_reward": total_reward / n,
        "avg_response_length": sum(all_lengths) / n,
        "avg_response_length_correct": correct_lengths / n_correct if n_correct else float("nan"),
        "avg_response_length_incorrect": incorrect_lengths / n_incorrect if n_incorrect else float("nan"),
        "avg_token_entropy": sum(all_entropies) / len(all_entropies) if all_entropies else float("nan"),
        "n_correct": n_correct,
        "n_total": n,
    }

    # --- pretty-print ---
    sep = "=" * 70
    print(f"\n{sep}\n[{label}] Generation log ({n} examples)\n{sep}")
    for i, rec in enumerate(records):
        print(f"\n--- Example {i+1} ---")
        print(f"  Prompt:        {rec['prompt'][:120]}")
        print(f"  Response:      {rec['response'][:200]}")
        print(f"  Ground truth:  {rec['ground_truth']}")
        print(f"  Reward:        {rec['reward']:.3f}  "
              f"(format={rec['format_reward']:.1f}, answer={rec['answer_reward']:.1f})")
        print(f"  Length:        {rec['response_length']} tokens  |  "
              f"Entropy: {rec['avg_token_entropy']}")
    print(f"\n{sep}\n[{label}] Aggregate metrics")
    for k, v in metrics.items():
        print(f"  {k:<40} {v}")
    print(sep)

    return {"records": records, "metrics": metrics}
