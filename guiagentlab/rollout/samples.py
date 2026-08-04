"""Batch-level contracts for multi-action rollout samples."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_rollout_group_semantics(
    non_tensor_batch: dict[str, Any],
    reward_rows: Any,
    *,
    group_size: int,
    invalid_action_penalty: float,
    advantage_estimator: str = "grpo",
) -> None:
    """Fail fast if task, trajectory, or per-action rewards were mixed."""
    required = {
        "uid",
        "trajectory_uid",
        "task_name",
        "episode_reward",
        "action_valid",
        "action_reward_adjustment",
    }
    missing = required.difference(non_tensor_batch)
    if missing:
        raise KeyError(f"rollout group audit is missing fields: {sorted(missing)}")
    uids = np.asarray(non_tensor_batch["uid"], dtype=object)
    trajectories = np.asarray(non_tensor_batch["trajectory_uid"], dtype=object)
    tasks = np.asarray(non_tensor_batch["task_name"], dtype=object)
    episode_rewards = np.asarray(non_tensor_batch["episode_reward"], dtype=float)
    action_valid = np.asarray(non_tensor_batch["action_valid"], dtype=bool)
    action_adjustments = np.asarray(
        non_tensor_batch["action_reward_adjustment"],
        dtype=float,
    )
    if hasattr(reward_rows, "detach"):
        scores = reward_rows.detach().sum(dim=-1).cpu().numpy()
    else:
        scores = np.asarray(reward_rows, dtype=float)
        if scores.ndim > 1:
            scores = scores.sum(axis=-1)
    lengths = {
        len(uids),
        len(trajectories),
        len(tasks),
        len(action_valid),
        len(action_adjustments),
        len(scores),
    }
    if len(lengths) != 1:
        raise ValueError(f"rollout audit fields have inconsistent lengths: {sorted(lengths)}")

    for trajectory in set(trajectories.tolist()):
        positions = np.flatnonzero(trajectories == trajectory)
        if len(set(uids[positions].tolist())) != 1:
            raise ValueError(f"trajectory {trajectory!r} crossed prompt groups")
        if len(set(tasks[positions].tolist())) != 1:
            raise ValueError(f"trajectory {trajectory!r} crossed tasks")
        if not np.allclose(episode_rewards[positions], episode_rewards[positions[0]]):
            raise ValueError(f"trajectory {trajectory!r} has inconsistent episode rewards")

    for uid in set(uids.tolist()):
        positions = np.flatnonzero(uids == uid)
        trajectory_count = len(set(trajectories[positions].tolist()))
        if trajectory_count != group_size:
            raise ValueError(
                f"prompt group {uid!r} has {trajectory_count} trajectories, "
                f"expected {group_size}"
            )
        if len(set(tasks[positions].tolist())) != 1:
            raise ValueError(f"prompt group {uid!r} mixed different tasks")

    if str(advantage_estimator).lower() == "admire_grpo":
        expected_scores = episode_rewards + action_adjustments
    else:
        expected_scores = episode_rewards - (~action_valid).astype(float) * float(
            invalid_action_penalty
        )
    if not np.allclose(scores, expected_scores, atol=1e-6):
        bad = np.flatnonzero(~np.isclose(scores, expected_scores, atol=1e-6))[:5]
        raise ValueError(f"per-action reward semantics differ at rows {bad.tolist()}")
    expected_adjustments = expected_scores - episode_rewards
    if not np.allclose(action_adjustments, expected_adjustments, atol=1e-6):
        bad = np.flatnonzero(
            ~np.isclose(action_adjustments, expected_adjustments, atol=1e-6)
        )[:5]
        raise ValueError(
            f"per-action reward adjustments differ at rows {bad.tolist()}"
        )
