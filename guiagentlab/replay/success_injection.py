"""Select all-failure rollout groups for successful trajectory injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReplayReplacement:
    prompt_uid: Any
    task_name: str
    trajectory_uid: str
    priority: int
    rows: np.ndarray


@dataclass(frozen=True)
class ReplaySelection:
    replacements: tuple[ReplayReplacement, ...]
    all_failure_groups: int
    ineligible_source_groups: int


def select_all_failure_replacements(
    non_tensor_batch: dict[str, Any],
    *,
    group_size: int,
    eligible_tasks: set[str] | frozenset[str],
) -> ReplaySelection:
    """Select the final trajectory from each eligible all-failure prompt group."""
    required = {"uid", "trajectory_uid", "task_name", "episode_reward", "priority"}
    missing = required.difference(non_tensor_batch)
    if missing:
        raise KeyError(f"Success replay rollout output is missing fields: {sorted(missing)}")

    uids = np.asarray(non_tensor_batch["uid"], dtype=object)
    trajectories = np.asarray(non_tensor_batch["trajectory_uid"], dtype=object)
    tasks = np.asarray(non_tensor_batch["task_name"], dtype=object)
    rewards = np.asarray(non_tensor_batch["episode_reward"], dtype=float)
    priorities = np.asarray(non_tensor_batch["priority"], dtype=np.int64)

    replacements: list[ReplayReplacement] = []
    all_failure_groups = 0
    ineligible_source_groups = 0
    for uid in dict.fromkeys(uids.tolist()):
        group_rows = np.flatnonzero(uids == uid)
        group_trajectories = list(dict.fromkeys(trajectories[group_rows].tolist()))
        trajectory_rewards = []
        for trajectory in group_trajectories:
            rows = group_rows[trajectories[group_rows] == trajectory]
            values = rewards[rows]
            if not np.allclose(values, values[0]):
                raise ValueError(
                    f"Success replay trajectory has inconsistent rewards: {trajectory}"
                )
            trajectory_rewards.append(float(values[0]))
        if any(reward > 0 for reward in trajectory_rewards):
            continue
        all_failure_groups += 1
        if len(group_trajectories) != group_size:
            raise ValueError(
                f"Success replay group {uid!r} has {len(group_trajectories)} trajectories, "
                f"expected {group_size}"
            )
        task_values = set(tasks[group_rows].tolist())
        if len(task_values) != 1:
            raise ValueError(f"Success replay group {uid!r} mixes tasks")
        task_name = str(next(iter(task_values)))
        if task_name not in eligible_tasks:
            ineligible_source_groups += 1
            continue

        target_trajectory = group_trajectories[-1]
        rows = group_rows[trajectories[group_rows] == target_trajectory]
        replacements.append(
            ReplayReplacement(
                prompt_uid=uid,
                task_name=task_name,
                trajectory_uid=str(target_trajectory),
                priority=int(priorities[rows[0]]),
                rows=rows,
            )
        )

    return ReplaySelection(
        replacements=tuple(replacements),
        all_failure_groups=all_failure_groups,
        ineligible_source_groups=ineligible_source_groups,
    )
