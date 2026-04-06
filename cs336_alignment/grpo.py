from __future__ import annotations

from typing import Callable

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
