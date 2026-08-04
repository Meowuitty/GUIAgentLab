"""Step-level ADMIRE-GRPO advantages from composed rollout rewards."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from guiagentlab.methods.advantages import group_normalize


def admire_grpo_advantages(
    total_rewards,
    response_mask,
    prompt_groups: Sequence[object],
    *,
    normalize_std: bool = True,
):
    """Normalize composed step rewards across every action in a prompt group."""
    if total_rewards.ndim != 1:
        raise ValueError("ADMIRE total rewards must be one-dimensional")
    if response_mask.ndim != 2 or response_mask.shape[0] != len(total_rewards):
        raise ValueError("response_mask must have one row per ADMIRE reward")
    normalized = group_normalize(
        total_rewards,
        prompt_groups,
        normalize_std=normalize_std,
    )
    advantages = normalized.unsqueeze(-1) * response_mask
    return advantages, advantages


def verl_admire_grpo_advantage(
    token_level_rewards,
    response_mask,
    index,
    *,
    config: Any,
    batch: dict[str, Any],
    **_: Any,
):
    """VERL adapter that computes advantages directly from composed step R."""
    if index is None:
        raise KeyError("ADMIRE-GRPO batch is missing uid")
    if "admire_total_reward" not in batch:
        raise KeyError("ADMIRE-GRPO batch is missing admire_total_reward")
    total_rewards = batch["admire_total_reward"].to(
        dtype=token_level_rewards.dtype,
        device=token_level_rewards.device,
    )
    normalize_std = bool(True if config is None else config.get("admire_normalize_std", True))
    return admire_grpo_advantages(
        total_rewards,
        response_mask,
        index,
        normalize_std=normalize_std,
    )
