"""Exclusive, failure-aware leasing for MobileWorld containers."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from guiagentlab.env.errors import FailureKind, InfrastructureError
from guiagentlab.env.recovery import EndpointRecovery


class ContainerRole(StrEnum):
    ACTIVE = "active"
    SPARE = "spare"


class ContainerState(StrEnum):
    AVAILABLE = "available"
    LEASED = "leased"
    QUARANTINED = "quarantined"
    RECOVERING = "recovering"
    RETIRED = "retired"


@dataclass(slots=True)
class Container:
    endpoint: str
    role: ContainerRole
    state: ContainerState = ContainerState.AVAILABLE
    lease_token: str | None = None
    failures: int = 0
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class Lease:
    endpoint: str
    token: str
    role: ContainerRole


class ContainerPool:
    """Thread-safe central pool; an endpoint can have at most one live lease."""

    def __init__(
        self,
        endpoints: list[str],
        *,
        active: int,
        spares: int,
        recovery: EndpointRecovery | None = None,
    ) -> None:
        if active <= 0 or spares < 0:
            raise ValueError("active must be positive and spares non-negative")
        expected = active + spares
        if len(endpoints) < expected:
            raise ValueError(f"expected at least {expected} endpoints, got {len(endpoints)}")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("container endpoints must be unique")
        endpoints = endpoints[:expected]
        self.recovery = recovery
        self._condition = threading.Condition()
        self._containers = {
            endpoint.rstrip("/"): Container(
                endpoint.rstrip("/"),
                ContainerRole.ACTIVE if i < active else ContainerRole.SPARE,
            )
            for i, endpoint in enumerate(endpoints)
        }
        if self.recovery is not None:
            persisted = self.recovery.statuses(list(self._containers))
            for endpoint, record in persisted.items():
                container = self._containers[endpoint]
                container.state = ContainerState.QUARANTINED
                container.failures = int(record.get("failures", 0))
                container.last_error = str(record.get("reason") or "persisted quarantine")
            for endpoint in persisted:
                self._start_recovery(endpoint)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        active: int,
        spares: int,
        recovery: EndpointRecovery | None = None,
    ) -> ContainerPool:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        endpoints = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return cls(
            endpoints,
            active=active,
            spares=spares,
            recovery=recovery,
        )

    def acquire(self, *, timeout: float | None = None) -> Lease:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                active_capacity = sum(
                    item.role == ContainerRole.ACTIVE for item in self._containers.values()
                )
                recovering = any(
                    item.state == ContainerState.RECOVERING
                    for item in self._containers.values()
                )
                serviceable = sum(
                    item.state in {ContainerState.AVAILABLE, ContainerState.LEASED}
                    for item in self._containers.values()
                )
                if serviceable < active_capacity and not recovering:
                    raise InfrastructureError(
                        FailureKind.NO_CAPACITY,
                        "healthy MobileWorld capacity is below the configured active count",
                        details={
                            "active_target": active_capacity,
                            "serviceable": serviceable,
                        },
                    )
                leased_total = sum(
                    item.state == ContainerState.LEASED for item in self._containers.values()
                )
                if serviceable >= active_capacity and leased_total < active_capacity:
                    for container in self._containers.values():
                        if (
                            container.role == ContainerRole.ACTIVE
                            and container.state == ContainerState.AVAILABLE
                        ):
                            return self._lease_locked(container)

                unavailable_active = sum(
                    item.role == ContainerRole.ACTIVE
                    and item.state
                    in {
                        ContainerState.QUARANTINED,
                        ContainerState.RECOVERING,
                        ContainerState.RETIRED,
                    }
                    for item in self._containers.values()
                )
                leased_spares = sum(
                    item.role == ContainerRole.SPARE and item.state == ContainerState.LEASED
                    for item in self._containers.values()
                )
                if (
                    serviceable >= active_capacity
                    and leased_total < active_capacity
                    and leased_spares < unavailable_active
                ):
                    for container in self._containers.values():
                        if (
                            container.role == ContainerRole.SPARE
                            and container.state == ContainerState.AVAILABLE
                        ):
                            return self._lease_locked(container)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise InfrastructureError(
                        FailureKind.NO_CAPACITY,
                        "timed out waiting for a healthy container",
                    )
                self._condition.wait(remaining)

    @staticmethod
    def _lease_locked(container: Container) -> Lease:
        token = uuid.uuid4().hex
        container.state = ContainerState.LEASED
        container.lease_token = token
        return Lease(container.endpoint, token, container.role)

    def release(self, lease: Lease) -> None:
        with self._condition:
            container = self._validate_lease_locked(lease)
            container.state = ContainerState.AVAILABLE
            container.lease_token = None
            self._condition.notify()

    def quarantine(self, lease: Lease, error: BaseException | str) -> None:
        endpoint: str
        with self._condition:
            container = self._validate_lease_locked(lease)
            container.state = ContainerState.QUARANTINED
            container.lease_token = None
            container.failures += 1
            container.last_error = str(error)
            endpoint = container.endpoint
            self._condition.notify_all()
        if self.recovery is not None:
            self.recovery.mark(endpoint, "quarantined", str(error))
            self._start_recovery(endpoint)

    def recover(self, endpoint: str) -> None:
        normalized = endpoint.rstrip("/")
        with self._condition:
            container = self._containers[normalized]
            if container.state == ContainerState.LEASED:
                raise ValueError("cannot recover a leased container")
            container.state = ContainerState.AVAILABLE
            container.last_error = None
            self._condition.notify()
        if self.recovery is not None:
            self.recovery.mark(normalized, "available", None)

    def retire(self, endpoint: str, reason: str) -> None:
        normalized = endpoint.rstrip("/")
        with self._condition:
            container = self._containers[normalized]
            if container.state == ContainerState.LEASED:
                raise ValueError("cannot retire a leased container")
            container.state = ContainerState.RETIRED
            container.last_error = reason
            self._condition.notify_all()
        if self.recovery is not None:
            self.recovery.mark(normalized, "retired", reason)

    def snapshot(self) -> list[dict[str, object]]:
        with self._condition:
            return [
                {
                    "endpoint": item.endpoint,
                    "role": item.role.value,
                    "state": item.state.value,
                    "failures": item.failures,
                    "last_error": item.last_error,
                }
                for item in self._containers.values()
            ]

    def _validate_lease_locked(self, lease: Lease) -> Container:
        try:
            container = self._containers[lease.endpoint]
        except KeyError as exc:
            raise ValueError("lease belongs to another pool") from exc
        if container.state != ContainerState.LEASED or container.lease_token != lease.token:
            raise ValueError("stale or invalid lease")
        return container

    def _start_recovery(self, endpoint: str) -> None:
        if self.recovery is None:
            return
        with self._condition:
            container = self._containers[endpoint]
            if container.state == ContainerState.RECOVERING:
                return
            if container.state == ContainerState.LEASED:
                raise ValueError("cannot recover a leased container")
            container.state = ContainerState.RECOVERING
            self._condition.notify_all()
        thread = threading.Thread(
            target=self._recover_endpoint,
            args=(endpoint,),
            name=f"endpoint-recovery-{endpoint.rsplit(':', 1)[-1]}",
            daemon=True,
        )
        thread.start()

    def _recover_endpoint(self, endpoint: str) -> None:
        assert self.recovery is not None
        try:
            self.recovery.recover_endpoint(endpoint)
        except BaseException as exc:
            with self._condition:
                container = self._containers[endpoint]
                if container.state == ContainerState.RECOVERING:
                    container.state = ContainerState.QUARANTINED
                    container.last_error = f"recovery failed: {exc}"
                    self._condition.notify_all()
            return
        with self._condition:
            container = self._containers[endpoint]
            if container.state == ContainerState.RECOVERING:
                container.state = ContainerState.AVAILABLE
                container.last_error = None
                self._condition.notify_all()
