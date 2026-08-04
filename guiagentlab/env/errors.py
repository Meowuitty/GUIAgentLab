"""Typed failures that must not be converted into model rewards."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FailureKind(StrEnum):
    HEALTH = "health"
    INITIALIZATION = "initialization"
    STEP = "step"
    EVALUATION = "evaluation"
    TEARDOWN = "teardown"
    TIMEOUT = "timeout"
    PROTOCOL = "protocol"
    NO_CAPACITY = "no_capacity"


@dataclass(slots=True)
class InfrastructureError(RuntimeError):
    kind: FailureKind
    message: str
    endpoint: str | None = None
    retryable: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        location = f" at {self.endpoint}" if self.endpoint else ""
        return f"{self.kind.value}{location}: {self.message}"
