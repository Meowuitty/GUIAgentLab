"""Cheap deterministic detection of repeated GUI actions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def normalized_action_key(
    action: dict[str, Any],
    *,
    coordinate_bucket: int = 25,
) -> tuple[object, ...]:
    """Return a jitter-resistant, hashable action representation."""
    if coordinate_bucket <= 0:
        raise ValueError("coordinate_bucket must be positive")

    action_type = str(action.get("action_type", "")).casefold()

    def bucket(field: str) -> int | None:
        value = action.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return int(value) // coordinate_bucket

    if action_type in {"click", "double_tap", "long_press"}:
        return action_type, bucket("x"), bucket("y")
    if action_type == "drag":
        return (
            action_type,
            bucket("start_x"),
            bucket("start_y"),
            bucket("end_x"),
            bucket("end_y"),
        )
    if action_type == "scroll":
        # Gesture coordinates vary between generations but rarely change the
        # meaning of a directional scroll.
        return action_type, str(action.get("direction", "")).casefold()
    if action_type in {"input_text", "answer", "ask_user"}:
        text = " ".join(str(action.get("text", "")).split())
        return action_type, text
    if action_type == "open_app":
        return action_type, str(action.get("app_name", "")).casefold()
    if action_type == "status":
        return action_type, str(action.get("goal_status", "")).casefold()
    return (action_type,)


@dataclass(slots=True)
class DeterministicLoopGuard:
    """Identify high-confidence repeats without asking the external PRM.

    A step is blocked when the same refined state/action pair occurs for the
    third time, or when a state/action sequence of length 2..N completes its
    second identical cycle.
    """

    repeat_threshold: int = 3
    max_cycle_length: int = 8
    coordinate_bucket: int = 25
    _counts: Counter[tuple[object, ...]] = field(default_factory=Counter, init=False)
    _history: list[tuple[object, ...]] = field(default_factory=list, init=False)
    _last_repeated_indices: tuple[int, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if self.repeat_threshold < 2:
            raise ValueError("repeat_threshold must be at least two")
        if self.max_cycle_length < 0:
            raise ValueError("max_cycle_length must be non-negative")
        if self.coordinate_bucket <= 0:
            raise ValueError("coordinate_bucket must be positive")

    def observe(
        self,
        *,
        anchor_uid: int,
        anchor_detail_uid: int,
        action: dict[str, Any],
    ) -> str | None:
        """Record a step and return a human-readable block reason, if any."""
        self._last_repeated_indices = ()
        action_key = normalized_action_key(
            action,
            coordinate_bucket=self.coordinate_bucket,
        )
        token = (int(anchor_uid), int(anchor_detail_uid), *action_key)
        self._counts[token] += 1
        self._history.append(token)

        count = self._counts[token]
        if count >= self.repeat_threshold:
            self._last_repeated_indices = (len(self._history) - 1,)
            return f"deterministic loop guard: refined state/action pair repeated {count} times"

        maximum = min(self.max_cycle_length, len(self._history) // 2)
        for cycle_length in range(2, maximum + 1):
            if self._history[-cycle_length:] == self._history[-2 * cycle_length : -cycle_length]:
                self._last_repeated_indices = tuple(
                    range(len(self._history) - cycle_length, len(self._history))
                )
                return (
                    "deterministic loop guard: state/action sequence of length "
                    f"{cycle_length} repeated"
                )
        return None

    @property
    def last_repeated_indices(self) -> tuple[int, ...]:
        """Indices in the just-completed repeated segment.

        The indices let callers suppress duplicate downstream work without
        changing the guard's existing string-returning API.
        """
        return self._last_repeated_indices
