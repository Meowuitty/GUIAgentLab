"""GiGPO episode- and anchor-group relative advantage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from guiagentlab.methods.advantages import group_normalize, trajectory_outcome_scores
from guiagentlab.rollout.maiui import fingerprint_hamming_distance


def perceptual_anchor_groups(
    anchor_signatures: Sequence[object],
    episode_groups: Sequence[object],
    *,
    detail_signatures: Sequence[object] | None = None,
    max_hamming_distance: int = 4,
    max_detail_hamming_distance: int = 3,
) -> list[object]:
    """Deterministically cluster near-identical GUI fingerprints within tasks.

    A coarse whole-screen hash provides noise tolerance. When detail signatures
    are supplied, a tiled hash must also match, preventing small but meaningful
    controls or text from being merged into the same anchor. Exact coarse-hash
    collisions use a stricter one-bit detail limit because any remaining local
    difference is disproportionately likely to be semantic.

    Exact signature pairs are collapsed before fuzzy clustering. Fuzzy clusters
    use an all-member rule: a signature may join only when it matches every
    existing member. Sorting the unique signatures makes the partition
    independent of rollout row order and prevents representative-point chaining
    from merging members that do not match each other.
    """
    if len(anchor_signatures) != len(episode_groups):
        raise ValueError("anchor signatures and episode groups must have equal length")
    if detail_signatures is not None and len(detail_signatures) != len(anchor_signatures):
        raise ValueError("detail signatures must match anchor signatures")
    if not 0 <= max_hamming_distance <= 64:
        raise ValueError("max_hamming_distance must be between 0 and 64")
    if not 0 <= max_detail_hamming_distance <= 64:
        raise ValueError("max_detail_hamming_distance must be between 0 and 64")
    positions: dict[object, list[int]] = defaultdict(list)
    for index, group in enumerate(episode_groups):
        positions[group].append(index)

    def distances(
        left: tuple[int, int | None],
        right: tuple[int, int | None],
    ) -> tuple[int, int]:
        coarse_distance = fingerprint_hamming_distance(left[0], right[0])
        detail_distance = (
            0
            if left[1] is None or right[1] is None
            else fingerprint_hamming_distance(left[1], right[1])
        )
        return coarse_distance, detail_distance

    def matches(
        left: tuple[int, int | None],
        right: tuple[int, int | None],
    ) -> bool:
        coarse_distance, detail_distance = distances(left, right)
        detail_limit = (
            min(max_detail_hamming_distance, 1)
            if coarse_distance == 0
            else max_detail_hamming_distance
        )
        return coarse_distance <= max_hamming_distance and detail_distance <= detail_limit

    output: list[object] = [None] * len(anchor_signatures)
    for episode_group, indices in positions.items():
        exact_positions: dict[tuple[int, int | None], list[int]] = defaultdict(list)
        for index in indices:
            signature = anchor_signatures[index]
            if not isinstance(signature, int):
                output[index] = (episode_group, signature)
                continue
            detail = None if detail_signatures is None else detail_signatures[index]
            if detail is not None and not isinstance(detail, int):
                output[index] = (episode_group, signature, detail)
                continue
            exact_positions[(signature, detail)].append(index)

        clusters: list[list[tuple[int, int | None]]] = []
        signature_clusters: dict[tuple[int, int | None], int] = {}
        ordered_signatures = sorted(
            exact_positions,
            key=lambda item: (item[0], item[1] is not None, item[1] or 0),
        )
        for signature in ordered_signatures:
            candidates = []
            for cluster_index, members in enumerate(clusters):
                member_distances = [distances(signature, member) for member in members]
                if all(matches(signature, member) for member in members):
                    candidates.append((max(member_distances), cluster_index))
            if candidates:
                _, cluster_index = min(candidates)
                clusters[cluster_index].append(signature)
            else:
                cluster_index = len(clusters)
                clusters.append([signature])
            signature_clusters[signature] = cluster_index

        for signature, signature_indices in exact_positions.items():
            cluster_index = signature_clusters[signature]
            for index in signature_indices:
                output[index] = (episode_group, "perceptual", cluster_index)
    return output


def _masked_group_normalize(
    scores,
    groups: Sequence[object],
    valid_rows,
    *,
    normalize_std: bool,
):
    """Use only valid rows for group statistics and return zero for masked rows."""
    import torch

    valid_rows = valid_rows.to(device=scores.device, dtype=torch.bool)
    if valid_rows.ndim != 1 or len(valid_rows) != len(scores):
        raise ValueError("valid_rows must match scores")
    output = torch.zeros_like(scores)
    indices = torch.nonzero(valid_rows, as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        return output
    selected_groups = [groups[index] for index in indices.cpu().tolist()]
    output[indices] = group_normalize(
        scores[indices],
        selected_groups,
        normalize_std=normalize_std,
    )
    return output


def _perceptual_anchor_groups_without_loops(
    anchor_groups: Sequence[object],
    prompt_groups: Sequence[object],
    loop_rows,
    *,
    anchor_detail_groups: Sequence[object] | None,
    max_hamming_distance: int,
    max_detail_hamming_distance: int,
) -> list[object]:
    """Cluster non-loop rows while assigning loop rows inert unique groups."""
    included_indices = [
        index for index, is_loop in enumerate(loop_rows.cpu().tolist()) if not is_loop
    ]
    included_groups = perceptual_anchor_groups(
        [anchor_groups[index] for index in included_indices],
        [prompt_groups[index] for index in included_indices],
        detail_signatures=(
            None
            if anchor_detail_groups is None
            else [anchor_detail_groups[index] for index in included_indices]
        ),
        max_hamming_distance=max_hamming_distance,
        max_detail_hamming_distance=max_detail_hamming_distance,
    )
    output: list[object] = [
        ("excluded-loop-anchor-row", index) for index in range(len(anchor_groups))
    ]
    for index, group in zip(included_indices, included_groups, strict=True):
        output[index] = group
    return output


def gigpo_advantages(
    episode_scores,
    step_returns,
    response_mask,
    prompt_groups,
    trajectory_groups,
    anchor_groups,
    *,
    anchor_detail_groups=None,
    process_reward_valid=None,
    loop_detected=None,
    action_reward_adjustments=None,
    local_advantages=None,
    step_weight: float = 1.0,
    outcome_normalize_std: bool = True,
    step_normalize_std: bool = False,
    anchor_hamming_distance: int = 4,
    anchor_detail_hamming_distance: int = 3,
):
    """Combine episode-relative and anchor-state-relative advantages.

    Each row represents one action step. ``prompt_groups`` joins alternative
    trajectories for the same task; ``trajectory_groups`` identifies repeated
    action rows from one trajectory; and ``anchor_groups`` joins actions generated
    from the same (or intentionally clustered) observation.
    """
    import torch

    if episode_scores.shape != step_returns.shape:
        raise ValueError("episode_scores and step_returns must have the same shape")
    if (
        len(prompt_groups) != len(episode_scores)
        or len(trajectory_groups) != len(episode_scores)
        or len(anchor_groups) != len(episode_scores)
    ):
        raise ValueError("group arrays must match the batch size")
    if response_mask.ndim != 2 or response_mask.shape[0] != len(episode_scores):
        raise ValueError("response_mask must have one row per action")

    if process_reward_valid is None:
        step_valid = torch.ones_like(episode_scores, dtype=torch.bool)
    elif isinstance(process_reward_valid, torch.Tensor):
        step_valid = process_reward_valid.to(
            dtype=torch.bool,
            device=episode_scores.device,
        )
    else:
        # VERL stores scalar non-tensor fields in NumPy object arrays. Torch
        # cannot consume an object ndarray directly even when every element is
        # a bool, so coerce the scalar values before tensor construction.
        step_valid = torch.tensor(
            [bool(value) for value in process_reward_valid],
            dtype=torch.bool,
            device=episode_scores.device,
        )
        if step_valid.ndim != 1 or len(step_valid) != len(episode_scores):
            raise ValueError("process_reward_valid must match the batch size")
    active_actions = response_mask.to(torch.bool).any(dim=-1)
    if loop_detected is None:
        loop_rows = torch.zeros_like(episode_scores, dtype=torch.bool)
    elif isinstance(loop_detected, torch.Tensor):
        loop_rows = loop_detected.to(
            dtype=torch.bool,
            device=episode_scores.device,
        )
    else:
        loop_rows = torch.tensor(
            [bool(value) for value in loop_detected],
            dtype=torch.bool,
            device=episode_scores.device,
        )
    if loop_rows.ndim != 1 or len(loop_rows) != len(episode_scores):
        raise ValueError("loop_detected must match the batch size")
    step_valid = step_valid & active_actions & ~loop_rows

    episode = trajectory_outcome_scores(
        episode_scores,
        prompt_groups,
        trajectory_groups,
        normalize_std=outcome_normalize_std,
    )
    if action_reward_adjustments is not None:
        if isinstance(action_reward_adjustments, torch.Tensor):
            adjustments = action_reward_adjustments.to(
                dtype=episode.dtype,
                device=episode.device,
            )
        else:
            adjustments = torch.tensor(
                [float(value) for value in action_reward_adjustments],
                dtype=episode.dtype,
                device=episode.device,
            )
        if adjustments.ndim != 1 or len(adjustments) != len(episode):
            raise ValueError("action_reward_adjustments must match the batch size")
        episode = episode + adjustments

    clustered_anchors = _perceptual_anchor_groups_without_loops(
        anchor_groups,
        prompt_groups,
        loop_rows,
        anchor_detail_groups=anchor_detail_groups,
        max_hamming_distance=anchor_hamming_distance,
        max_detail_hamming_distance=anchor_detail_hamming_distance,
    )
    step = _masked_group_normalize(
        step_returns,
        clustered_anchors,
        step_valid,
        normalize_std=step_normalize_std,
    )
    combined = episode + step_weight * step
    if local_advantages is not None:
        local = torch.as_tensor(
            local_advantages,
            dtype=combined.dtype,
            device=combined.device,
        )
        if local.ndim != 1 or len(local) != len(combined):
            raise ValueError("local_advantages must match the batch size")
        combined = combined + local
    advantages = combined.unsqueeze(-1) * response_mask
    return advantages, advantages


def verl_gigpo_advantage(
    token_level_rewards,
    response_mask,
    index,
    config=None,
    non_tensor_batch=None,
    batch=None,
    **_: Any,
):
    """verl adapter.

    The explicit verl integration patch passes ``non_tensor_batch`` and ``batch``
    to custom estimators. Required fields are emitted by the GiGPO data/rollout
    path and are validated here rather than silently approximated.
    """
    if non_tensor_batch is None or batch is None:
        raise RuntimeError(
            "GiGPO requires the GUIAgentLab verl custom-estimator data patch; "
            "non_tensor_batch/batch were not provided"
        )
    required = {
        "anchor_uid",
        "trajectory_uid",
        "episode_reward",
        "process_reward_valid",
        "action_reward_adjustment",
        "loop_detected",
        "successful_terminal",
    }
    missing = required.difference(non_tensor_batch)
    if missing or "step_returns" not in batch or index is None:
        all_missing = missing | ({"step_returns"} - set(batch))
        if index is None:
            all_missing.add("uid")
        raise KeyError(f"GiGPO batch is missing: {sorted(all_missing)}")

    import torch

    cfg = config or {}
    get = cfg.get if hasattr(cfg, "get") else lambda key, default: default
    episode_scores = torch.as_tensor(
        [float(value) for value in non_tensor_batch["episode_reward"]],
        dtype=token_level_rewards.dtype,
        device=token_level_rewards.device,
    )
    loop_advantage = float(get("gigpo_loop_advantage", -0.25))
    successful_terminal_advantage = float(get("gigpo_successful_terminal_advantage", 0.25))
    if loop_advantage > 0:
        raise ValueError("gigpo_loop_advantage must be non-positive")
    if successful_terminal_advantage < 0:
        raise ValueError("gigpo_successful_terminal_advantage must be non-negative")
    local_advantages = torch.zeros_like(episode_scores)
    loop_rows = torch.as_tensor(
        [bool(value) for value in non_tensor_batch["loop_detected"]],
        dtype=torch.bool,
        device=episode_scores.device,
    )
    successful_terminal_rows = torch.as_tensor(
        [bool(value) for value in non_tensor_batch["successful_terminal"]],
        dtype=torch.bool,
        device=episode_scores.device,
    )
    local_advantages[loop_rows] += loop_advantage
    local_advantages[successful_terminal_rows] += successful_terminal_advantage
    return gigpo_advantages(
        episode_scores,
        batch["step_returns"],
        response_mask,
        index,
        non_tensor_batch["trajectory_uid"],
        non_tensor_batch["anchor_uid"],
        anchor_detail_groups=non_tensor_batch.get("anchor_detail_uid"),
        process_reward_valid=non_tensor_batch["process_reward_valid"],
        loop_detected=loop_rows,
        action_reward_adjustments=non_tensor_batch["action_reward_adjustment"],
        local_advantages=local_advantages,
        step_weight=float(get("gigpo_step_weight", 1.0)),
        outcome_normalize_std=bool(get("gigpo_outcome_normalize_std", True)),
        step_normalize_std=bool(get("gigpo_step_normalize_std", False)),
        anchor_hamming_distance=int(get("gigpo_anchor_hamming_distance", 4)),
        anchor_detail_hamming_distance=int(get("gigpo_anchor_detail_hamming_distance", 3)),
    )
