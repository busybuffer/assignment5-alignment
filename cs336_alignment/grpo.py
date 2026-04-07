from __future__ import annotations

from typing import Callable, Literal

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute rewards for each group of rollout responses, normalized by the group size.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], scores the rollout responses
            against the ground truths, producing a dict with keys "reward",
            "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. Length is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str], ground truths repeated group_size times each.
            Length is rollout_batch_size.
        group_size: int, number of responses per question (group).
        advantage_eps: float, small constant to avoid division by zero.
        normalize_by_std: bool, if True divide by per-group std; otherwise subtract mean only.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            advantages: shape (rollout_batch_size,), group-normalized rewards.
            raw_rewards: shape (rollout_batch_size,), unnormalized rewards.
            metadata: dict of logged statistics.
    """
    rollout_batch_size = len(rollout_responses)
    n_groups = rollout_batch_size // group_size

    raw_rewards_list = [
        reward_fn(response, ground_truth)["reward"]
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    rewards_grouped = raw_rewards.view(n_groups, group_size)
    group_means = rewards_grouped.mean(dim=1, keepdim=True)

    if normalize_by_std:
        group_stds = rewards_grouped.std(dim=1, keepdim=True)
        advantages_grouped = (rewards_grouped - group_means) / (group_stds + advantage_eps)
    else:
        advantages_grouped = rewards_grouped - group_means

    advantages = advantages_grouped.view(rollout_batch_size)

    metadata = {
        "mean_reward": float(raw_rewards.mean()),
        "std_reward": float(raw_rewards.std()),
        "max_reward": float(raw_rewards.max()),
        "min_reward": float(raw_rewards.min()),
        "mean_advantage": float(advantages.mean()),
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Per-token policy-gradient loss (negative REINFORCE objective).

    Broadcasts scalar reward or advantage over the sequence dimension.
    """
    return -(raw_rewards_or_advantages * policy_log_probs)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-token GRPO/PPO clipped surrogate loss (negative clipped objective).

    advantages broadcasts over the sequence dimension.
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    surrogate_unclipped = ratio * advantages
    surrogate_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
    loss = -torch.min(surrogate_unclipped, surrogate_clipped)
    metadata: dict[str, torch.Tensor] = {
        "is_clipped": surrogate_clipped < surrogate_unclipped,
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the appropriate per-token policy-gradient loss."""
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for loss_type='no_baseline'"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    if loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages is required for loss_type='reinforce_with_baseline'"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    if loss_type == "grpo_clip":
        assert advantages is not None, "advantages is required for loss_type='grpo_clip'"
        assert old_log_probs is not None, "old_log_probs is required for loss_type='grpo_clip'"
        assert cliprange is not None, "cliprange is required for loss_type='grpo_clip'"
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
        return loss, metadata
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    length_norm: Literal["masked_mean", "masked_normalize"] = "masked_mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked per-token policy loss, scaled for gradient accumulation; calls ``backward``.

    length_norm controls how the per-token losses are aggregated across a sequence:
      - "masked_mean": divide by total response token count (equal weight per token).
      - "masked_normalize": divide by batch_size only (sum over tokens; longer responses
        contribute more gradient signal).
    """
    batch_size = policy_log_probs.shape[0]
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    m = response_mask.to(dtype=per_token_loss.dtype)
    if length_norm == "masked_mean":
        denom = m.sum().clamp(min=1e-8)
        aggregated = (per_token_loss * m).sum() / denom
    else:  # masked_normalize: sum tokens, normalize by sequence count only
        aggregated = (per_token_loss * m).sum() / batch_size
        denom = m.sum().clamp(min=1e-8)  # still used for clip_fraction below
    loss = aggregated / (batch_size * gradient_accumulation_steps)
    loss.backward()

    out_meta: dict[str, torch.Tensor] = dict(metadata)
    if "is_clipped" in metadata:
        ic = metadata["is_clipped"].to(dtype=per_token_loss.dtype)
        out_meta["clip_fraction"] = ((ic * m).sum() / denom).detach()

    return loss.detach(), out_meta
