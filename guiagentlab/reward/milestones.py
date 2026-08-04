"""Backend-verifiable milestone tracking and ADMIRE reward composition."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class MilestoneSnapshot:
    """One exact backend read of a task's milestone predicates."""

    registered: bool
    states: Mapping[str, bool]

    def __post_init__(self) -> None:
        states = dict(self.states)
        if not self.registered and states:
            raise ValueError("unregistered milestone snapshots must not contain states")
        if self.registered and not states:
            raise ValueError("registered milestone snapshots must contain at least one state")
        if any(not isinstance(key, str) or not key for key in states):
            raise ValueError("milestone ids must be non-empty strings")
        if any(type(value) is not bool for value in states.values()):
            raise ValueError("milestone states must be booleans")
        object.__setattr__(self, "states", MappingProxyType(states))

    @classmethod
    def from_payload(cls, payload: Any) -> MilestoneSnapshot:
        if not isinstance(payload, dict) or type(payload.get("registered")) is not bool:
            raise ValueError("milestone response must contain a boolean registered field")
        states = payload.get("milestones")
        if not isinstance(states, dict):
            raise ValueError("milestone response must contain a milestones object")
        return cls(registered=payload["registered"], states=states)


@dataclass(frozen=True, slots=True)
class MilestoneObservation:
    """Monotonic, baseline-excluded progress from an exact backend snapshot."""

    registered: bool
    completed: tuple[str, ...]
    newly_completed: tuple[str, ...]
    progress: float
    hit: float


class MilestoneTracker:
    """Track unordered milestone first hits without rewarding initial state."""

    def __init__(self, initial: MilestoneSnapshot) -> None:
        self._registered = initial.registered
        self._ids = tuple(initial.states)
        self._id_set = frozenset(self._ids)
        self._ever_completed = {
            milestone_id for milestone_id, completed in initial.states.items() if completed
        }
        self._initial_completed = frozenset(self._ever_completed)
        self._rewardable_count = len(self._id_set.difference(self._initial_completed))

    @property
    def registered(self) -> bool:
        return self._registered

    @property
    def milestone_ids(self) -> tuple[str, ...]:
        return self._ids

    @property
    def initial_completed(self) -> tuple[str, ...]:
        return tuple(
            milestone_id for milestone_id in self._ids if milestone_id in self._initial_completed
        )

    def observe(self, snapshot: MilestoneSnapshot) -> MilestoneObservation:
        if snapshot.registered != self._registered:
            raise ValueError("milestone registration changed during the episode")
        if not self._registered:
            return MilestoneObservation(
                registered=False,
                completed=(),
                newly_completed=(),
                progress=0.0,
                hit=0.0,
            )
        if frozenset(snapshot.states) != self._id_set:
            raise ValueError("milestone ids changed during the episode")

        currently_completed = {
            milestone_id for milestone_id, completed in snapshot.states.items() if completed
        }
        newly_completed_set = currently_completed.difference(self._ever_completed)
        self._ever_completed.update(currently_completed)
        credited_completed = tuple(
            milestone_id
            for milestone_id in self._ids
            if milestone_id in self._ever_completed and milestone_id not in self._initial_completed
        )
        newly_completed = tuple(
            milestone_id for milestone_id in self._ids if milestone_id in newly_completed_set
        )
        return MilestoneObservation(
            registered=True,
            completed=credited_completed,
            newly_completed=newly_completed,
            # Seeded facts are recorded for auditability but are not policy
            # progress and therefore must not dilute the denominator. Tasks
            # whose every predicate is baseline-true intentionally stay at 0.
            progress=(
                len(credited_completed) / self._rewardable_count
                if self._rewardable_count
                else 0.0
            ),
            # One environment action receives one critical-step credit even if
            # several independent backend predicates become true atomically.
            hit=float(bool(newly_completed)),
        )


def asymmetric_milestone_rewards(
    observations: Sequence[MilestoneObservation],
    *,
    successful: bool,
    failed_hit_bonus: float = 0.5,
) -> list[float]:
    """Apply ADMIRE's asymmetric positive/negative trajectory assignment."""
    if failed_hit_bonus < 0:
        raise ValueError("failed_hit_bonus must be non-negative")
    if successful:
        return [float(observation.hit) for observation in observations]
    return [
        float(observation.progress + failed_hit_bonus * observation.hit)
        for observation in observations
    ]


def curriculum_epoch(
    training_step: int,
    *,
    total_training_steps: int,
    total_epochs: int,
) -> int:
    """Map VERL's one-based global step to ADMIRE's zero-based epoch."""
    if training_step < 1:
        return 0
    if total_training_steps <= 0 or total_epochs <= 0:
        raise ValueError("total_training_steps and total_epochs must be positive")
    steps_per_epoch = max(1, math.ceil(total_training_steps / total_epochs))
    return min(total_epochs - 1, (training_step - 1) // steps_per_epoch)


def compose_admire_rewards(
    *,
    outcome: float,
    milestone_rewards: Sequence[float],
    action_valid: Sequence[bool],
    loop_detected: Sequence[bool],
    successful_terminal: Sequence[bool],
    epoch: int,
    milestone_weight: float = 0.3,
    milestone_decay: float = 0.99,
    invalid_coefficient: float = 0.5,
    invalid_reward: float = -1.0,
    loop_reward: float = -0.25,
    successful_terminal_reward: float = 0.25,
) -> tuple[list[float], float]:
    """Compose outcome, format, milestone, and existing local step rewards."""
    lengths = {
        len(milestone_rewards),
        len(action_valid),
        len(loop_detected),
        len(successful_terminal),
    }
    if len(lengths) != 1:
        raise ValueError("all ADMIRE step fields must have equal length")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if milestone_weight < 0 or not 0 <= milestone_decay <= 1:
        raise ValueError("milestone weight/decay is invalid")
    milestone_coefficient = milestone_weight * milestone_decay**epoch
    totals = []
    for process, valid, loop, terminal in zip(
        milestone_rewards,
        action_valid,
        loop_detected,
        successful_terminal,
        strict=True,
    ):
        format_component = 0.0 if valid else invalid_coefficient * invalid_reward
        local_component = (loop_reward if loop else 0.0) + (
            successful_terminal_reward if terminal else 0.0
        )
        totals.append(
            float(
                outcome
                + format_component
                + milestone_coefficient * float(process)
                + local_component
            )
        )
    return totals, milestone_coefficient
