"""Conservative teacher-intervention state used by OPD rollout."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ConservativeTeacherIntervention:
    """Track a small, auditable budget of teacher rescue calls.

    A call consumes budget before its result is known.  This intentionally
    counts rejected and failed teacher proposals as interventions.
    """

    enabled: bool = False
    max_calls: int = 2
    cooldown_steps: int = 2
    no_change_threshold: int = 3
    calls: int = 0
    applications: int = 0
    cooldown_remaining: int = 0

    def __post_init__(self) -> None:
        if self.max_calls < 0:
            raise ValueError("teacher max_calls must be non-negative")
        if self.cooldown_steps < 0:
            raise ValueError("teacher cooldown_steps must be non-negative")
        if self.no_change_threshold < 1:
            raise ValueError("teacher no_change_threshold must be positive")

    def start_step(self) -> bool:
        """Advance cooldown and return whether this step is protected by it."""
        in_cooldown = self.cooldown_remaining > 0
        if in_cooldown:
            self.cooldown_remaining -= 1
        return in_cooldown

    def trigger_reasons(
        self,
        *,
        action_valid: bool,
        loop_reason: str | None,
        no_change_streak: int,
        in_cooldown: bool,
    ) -> tuple[str, ...]:
        """Return deterministic reasons for calling the teacher on this step."""
        if (
            not self.enabled
            or self.calls >= self.max_calls
            or in_cooldown
        ):
            return ()

        reasons: list[str] = []
        if not action_valid:
            reasons.append("invalid_action")
        if loop_reason is not None:
            reasons.append("deterministic_loop")
        if no_change_streak >= self.no_change_threshold:
            reasons.append("no_state_change")
        return tuple(reasons)

    def record_call(self) -> None:
        """Consume one intervention slot before awaiting the teacher."""
        if not self.enabled:
            raise RuntimeError("cannot record a teacher call while intervention is disabled")
        if self.calls >= self.max_calls:
            raise RuntimeError("teacher intervention budget is exhausted")
        self.calls += 1

    def record_application(self) -> None:
        """Record an accepted teacher action and start the cooldown."""
        if self.applications >= self.calls:
            raise RuntimeError("cannot apply a teacher action without a recorded call")
        self.applications += 1
        self.cooldown_remaining = self.cooldown_steps
