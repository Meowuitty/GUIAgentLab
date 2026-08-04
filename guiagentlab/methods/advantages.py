"""Trajectory-aware GRPO primitives shared by GRPO and GiGPO."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def group_normalize(
    scores,
    groups: Sequence[object],
    *,
    normalize_std: bool = True,
    epsilon: float = 1e-6,
):
    """Normalize a 1-D torch tensor within prompt groups.

    Singleton groups receive zero advantage. This avoids treating an absolute
    score as a relative advantage when no comparison sample exists.
    """
    import torch

    if scores.ndim != 1 or len(scores) != len(groups):
        raise ValueError("scores and groups must have the same one-dimensional length")
    positions: dict[object, list[int]] = defaultdict(list)
    for position, group in enumerate(groups):
        positions[group].append(position)

    output = torch.zeros_like(scores)
    with torch.no_grad():
        for indices in positions.values():
            values = scores[indices]
            centered = values - values.mean()
            if normalize_std and len(indices) > 1:
                centered = centered / (values.std(unbiased=True) + epsilon)
            output[indices] = centered
    return output


def outcome_advantages(token_rewards, response_mask, groups, *, normalize_std: bool = True):
    scores = token_rewards.sum(dim=-1)
    normalized = group_normalize(scores, groups, normalize_std=normalize_std)
    advantages = normalized.unsqueeze(-1) * response_mask
    return advantages, advantages


def trajectory_outcome_scores(
    episode_scores,
    prompt_groups: Sequence[object],
    trajectory_groups: Sequence[object],
    *,
    normalize_std: bool,
):
    """Normalize one outcome per trajectory, then broadcast it to action rows."""
    if episode_scores.ndim != 1:
        raise ValueError("episode_scores must be one-dimensional")
    if len(prompt_groups) != len(episode_scores) or len(trajectory_groups) != len(
        episode_scores
    ):
        raise ValueError("trajectory outcome fields must match the batch size")

    trajectory_positions: dict[object, list[int]] = defaultdict(list)
    for position, trajectory in enumerate(trajectory_groups):
        trajectory_positions[trajectory].append(position)

    trajectory_scores = []
    trajectory_prompts = []
    trajectory_order = []
    for trajectory, positions in trajectory_positions.items():
        prompts = {prompt_groups[position] for position in positions}
        if len(prompts) != 1:
            raise ValueError(f"trajectory {trajectory!r} crossed prompt groups")
        values = episode_scores[positions]
        if not bool((values == values[0]).all()):
            raise ValueError(f"trajectory {trajectory!r} has inconsistent episode scores")
        trajectory_order.append(trajectory)
        trajectory_scores.append(values[0])
        trajectory_prompts.append(next(iter(prompts)))

    import torch

    unique_scores = torch.stack(trajectory_scores)
    normalized = group_normalize(
        unique_scores,
        trajectory_prompts,
        normalize_std=normalize_std,
    )
    by_trajectory = dict(zip(trajectory_order, normalized, strict=True))
    return torch.stack([by_trajectory[trajectory] for trajectory in trajectory_groups])


def hierarchical_action_weights(
    prompt_groups: Sequence[object],
    trajectory_groups: Sequence[object],
    valid_rows,
    *,
    normalizer: int | None = None,
):
    """Return action weights for group→trajectory→action mean aggregation.

    VERL's ``seq-mean-token-mean`` performs token→action aggregation. Multiplying
    each action loss by this positive weight makes its subsequent row mean equal
    to an action→trajectory→prompt-group hierarchy.
    """
    import torch

    if len(prompt_groups) != len(trajectory_groups):
        raise ValueError("prompt and trajectory groups must have equal length")
    valid_rows = torch.as_tensor(valid_rows, dtype=torch.bool)
    if valid_rows.ndim != 1 or len(valid_rows) != len(prompt_groups):
        raise ValueError("valid_rows must match group fields")

    trajectory_prompt: dict[object, object] = {}
    actions_per_trajectory: dict[object, int] = defaultdict(int)
    trajectories_per_prompt: dict[object, set[object]] = defaultdict(set)
    for position, (prompt, trajectory) in enumerate(
        zip(prompt_groups, trajectory_groups, strict=True)
    ):
        previous = trajectory_prompt.setdefault(trajectory, prompt)
        if previous != prompt:
            raise ValueError(f"trajectory {trajectory!r} crossed prompt groups")
        if bool(valid_rows[position]):
            actions_per_trajectory[trajectory] += 1
            trajectories_per_prompt[prompt].add(trajectory)

    active_prompts = {
        prompt: trajectories
        for prompt, trajectories in trajectories_per_prompt.items()
        if trajectories
    }
    if not active_prompts:
        raise ValueError("hierarchical loss batch contains no valid action rows")
    scale = int(normalizer) if normalizer is not None else int(valid_rows.sum().item())
    if scale <= 0:
        raise ValueError("hierarchical loss normalizer must be positive")

    weights = torch.zeros(len(prompt_groups), dtype=torch.float32, device=valid_rows.device)
    prompt_count = len(active_prompts)
    for position, (prompt, trajectory) in enumerate(
        zip(prompt_groups, trajectory_groups, strict=True)
    ):
        if not bool(valid_rows[position]):
            continue
        weights[position] = scale / (
            prompt_count * len(active_prompts[prompt]) * actions_per_trajectory[trajectory]
        )
    return weights


def trajectory_outcome_advantages(
    episode_scores,
    response_mask,
    prompt_groups: Sequence[object],
    trajectory_groups: Sequence[object],
    *,
    action_reward_adjustments=None,
    normalize_std: bool = True,
):
    """Compute GUI GRPO advantage without weighting outcomes by action count."""
    import torch

    if response_mask.ndim != 2 or response_mask.shape[0] != len(episode_scores):
        raise ValueError("response_mask must have one row per action")
    normalized = trajectory_outcome_scores(
        episode_scores,
        prompt_groups,
        trajectory_groups,
        normalize_std=normalize_std,
    )
    if action_reward_adjustments is not None:
        if isinstance(action_reward_adjustments, torch.Tensor):
            adjustments = action_reward_adjustments.to(
                dtype=normalized.dtype,
                device=normalized.device,
            )
        else:
            adjustments = torch.tensor(
                [float(value) for value in action_reward_adjustments],
                dtype=normalized.dtype,
                device=normalized.device,
            )
        if adjustments.ndim != 1 or len(adjustments) != len(normalized):
            raise ValueError("action_reward_adjustments must match the batch size")
        normalized = normalized + adjustments
    advantages = normalized.unsqueeze(-1) * response_mask
    return advantages, advantages


def verl_trajectory_grpo_advantage(
    token_level_rewards,
    response_mask,
    index,
    *,
    non_tensor_batch,
    normalize_std: bool = True,
):
    """Adapter for GUI multi-action trajectories using VERL's GRPO branch."""
    required = {"trajectory_uid", "episode_reward", "action_reward_adjustment"}
    missing = required.difference(non_tensor_batch)
    if missing or index is None:
        if index is None:
            missing.add("uid")
        raise KeyError(f"trajectory GRPO batch is missing: {sorted(missing)}")

    import torch

    episode_scores = torch.as_tensor(
        [float(value) for value in non_tensor_batch["episode_reward"]],
        dtype=token_level_rewards.dtype,
        device=token_level_rewards.device,
    )
    return trajectory_outcome_advantages(
        episode_scores,
        response_mask,
        index,
        non_tensor_batch["trajectory_uid"],
        action_reward_adjustments=non_tensor_batch["action_reward_adjustment"],
        normalize_std=normalize_std,
    )
