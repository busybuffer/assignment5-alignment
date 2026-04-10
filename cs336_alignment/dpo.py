from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from cs336_alignment.data import ALPACA_TEMPLATE


def _batch_sequence_log_probs(
    model: torch.nn.Module,
    sequences: list[torch.Tensor],
    no_grad: bool = False,
) -> torch.Tensor:
    """
    Compute sum of token log-probs for multiple sequences in one forward pass.

    Sequences are right-padded to the same length. A mask ensures padding
    positions do not contribute to the log-prob sums.

    Args:
        model:     language model
        sequences: list of 1-D LongTensors, all on the same device
        no_grad:   if True, wrap forward in torch.no_grad()

    Returns:
        1-D FloatTensor of shape (N,) — one log-prob sum per sequence.
    """
    device = sequences[0].device
    max_len = max(s.shape[0] for s in sequences)
    N = len(sequences)

    padded = torch.zeros(N, max_len, dtype=torch.long, device=device)
    attention_mask = torch.zeros(N, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        L = seq.shape[0]
        padded[i, :L] = seq
        attention_mask[i, :L] = 1

    if no_grad:
        with torch.no_grad():
            logits = model(padded, attention_mask=attention_mask).logits
    else:
        logits = model(padded, attention_mask=attention_mask).logits  # (N, T, V)

    # Shift: position t predicts token t+1
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)   # (N, T-1, V)
    target_ids = padded[:, 1:]                               # (N, T-1)
    token_log_probs = log_probs.gather(
        2, target_ids.unsqueeze(-1)
    ).squeeze(-1)                                            # (N, T-1)

    # Only sum over real (non-padding) target positions
    mask = attention_mask[:, 1:].float()                     # (N, T-1)
    return (token_log_probs * mask).sum(dim=-1)              # (N,)


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Compute the per-instance DPO loss (Equation 3).

    Uses the Alpaca template to format (prompt, response) pairs and appends
    the EOS token. The two models may be on different devices; the loss is
    returned on lm's device.

    Loss = -log σ(β * [(log πθ(yw|x) - log πref(yw|x))
                       - (log πθ(yl|x) - log πref(yl|x))])

    By the cancellation observation in the problem statement we compute
    unconditional sequence log-probs instead of conditional ones.
    Chosen and rejected are batched together for a single forward pass per model.
    """
    eos = tokenizer.eos_token or ""
    lm_device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device

    def tokenize(text: str, device: torch.device) -> torch.Tensor:
        """Returns a 1-D LongTensor of token ids."""
        ids = tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.long, device=device)

    chosen_text = ALPACA_TEMPLATE.format(
        instruction=prompt, response=response_chosen
    ) + eos
    rejected_text = ALPACA_TEMPLATE.format(
        instruction=prompt, response=response_rejected
    ) + eos

    # Policy forward: batch of 2, gradients flow through
    lm_log_probs = _batch_sequence_log_probs(
        lm,
        [tokenize(chosen_text, lm_device), tokenize(rejected_text, lm_device)],
        no_grad=False,
    )
    log_prob_chosen_lm = lm_log_probs[0]
    log_prob_rejected_lm = lm_log_probs[1]

    # Reference forward: batch of 2, no gradients needed
    ref_log_probs = _batch_sequence_log_probs(
        lm_ref,
        [tokenize(chosen_text, ref_device), tokenize(rejected_text, ref_device)],
        no_grad=True,
    ).to(lm_device)
    log_prob_chosen_ref = ref_log_probs[0]
    log_prob_rejected_ref = ref_log_probs[1]

    # DPO implicit reward difference
    reward_diff = beta * (
        (log_prob_chosen_lm - log_prob_chosen_ref)
        - (log_prob_rejected_lm - log_prob_rejected_ref)
    )

    return -F.logsigmoid(reward_diff)
