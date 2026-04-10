from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from cs336_alignment.data import ALPACA_TEMPLATE


def _sequence_log_prob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the sum of token log-probabilities for a sequence under `model`.

    input_ids: (1, T)  on whatever device the model is on.
    Returns a scalar tensor on the same device.
    """
    with torch.no_grad():
        logits = model(input_ids).logits  # (1, T, V)

    # Shift: predict token t+1 from position t
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T-1, V)
    target_ids = input_ids[:, 1:]                          # (1, T-1)

    # Gather the log-prob of each actual next token
    token_log_probs = log_probs.gather(
        2, target_ids.unsqueeze(-1)
    ).squeeze(-1)  # (1, T-1)

    return token_log_probs.sum()


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

    Loss = -log σ(β * (log πθ(yw|x) - log πref(yw|x))
                   - β * (log πθ(yl|x) - log πref(yl|x)))

    By the cancellation observation in the problem statement we compute
    unconditional sequence log-probs instead of conditional ones.
    """
    eos = tokenizer.eos_token or ""
    lm_device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device

    def tokenize(text: str, device: torch.device) -> torch.Tensor:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long, device=device)

    chosen_text = ALPACA_TEMPLATE.format(
        instruction=prompt, response=response_chosen
    ) + eos

    rejected_text = ALPACA_TEMPLATE.format(
        instruction=prompt, response=response_rejected
    ) + eos

    # Log-probs under the policy model (lm)
    log_prob_chosen_lm = _sequence_log_prob(lm, tokenize(chosen_text, lm_device))
    log_prob_rejected_lm = _sequence_log_prob(lm, tokenize(rejected_text, lm_device))

    # Log-probs under the reference model (lm_ref) — may be on a different device
    with torch.no_grad():
        log_prob_chosen_ref = _sequence_log_prob(
            lm_ref, tokenize(chosen_text, ref_device)
        ).to(lm_device)
        log_prob_rejected_ref = _sequence_log_prob(
            lm_ref, tokenize(rejected_text, ref_device)
        ).to(lm_device)

    # DPO implicit reward difference
    reward_diff = beta * (
        (log_prob_chosen_lm - log_prob_chosen_ref)
        - (log_prob_rejected_lm - log_prob_rejected_ref)
    )

    loss = -F.logsigmoid(reward_diff)
    return loss
