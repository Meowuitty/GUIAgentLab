"""Async state machine used as a single named Ray actor during training."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from guiagentlab.env.pool import ContainerRole, ContainerState
from guiagentlab.env.recovery import EndpointRecovery


@dataclass(slots=True)
class AsyncContainer:
    endpoint: str
    role: ContainerRole
    state: ContainerState = ContainerState.AVAILABLE
    token: str | None = None
    failures: int = 0
    last_error: str | None = None


class AsyncContainerPool:
    """Ray-actor implementation; methods are async so waiting never blocks releases."""

    def __init__(
        self,
        endpoints: list[str],
        active: int,
        spares: int,
        initialization_concurrency: int | None = None,
        teardown_concurrency: int | None = None,
        recovery_concurrency: int | None = None,
        recovery_config: dict[str, Any] | None = None,
    ):
        expected = active + spares
        if len(endpoints) < expected or len(set(endpoints)) != len(endpoints):
            raise ValueError(
                "endpoint count/uniqueness does not satisfy active + spares"
            )
        endpoints = endpoints[:expected]
        if initialization_concurrency is None:
            initialization_concurrency = min(4, active)
        if not 1 <= initialization_concurrency <= active:
            raise ValueError("initialization concurrency must be in [1, active]")
        if teardown_concurrency is None:
            teardown_concurrency = 1
        if not 1 <= teardown_concurrency <= active:
            raise ValueError("teardown concurrency must be in [1, active]")
        if recovery_concurrency is None:
            recovery_concurrency = min(2, active)
        if not 1 <= recovery_concurrency <= active:
            raise ValueError("recovery concurrency must be in [1, active]")
        self.recovery = EndpointRecovery.from_serializable(recovery_config)
        self.condition = asyncio.Condition()
        self._recovery_tasks: dict[str, asyncio.Task[None]] = {}
        self._recovery_semaphore = asyncio.Semaphore(recovery_concurrency)
        self.initialization_concurrency = initialization_concurrency
        self.teardown_concurrency = teardown_concurrency
        self._initializing: set[str] = set()
        self._tearing_down: set[str] = set()
        self.containers = {
            endpoint.rstrip("/"): AsyncContainer(
                endpoint.rstrip("/"),
                ContainerRole.ACTIVE if index < active else ContainerRole.SPARE,
            )
            for index, endpoint in enumerate(endpoints)
        }

    async def start(self) -> None:
        """Restore durable quarantine before workers can acquire endpoints."""
        if self.recovery is None:
            return
        persisted = await asyncio.to_thread(
            self.recovery.statuses, list(self.containers)
        )
        async with self.condition:
            for endpoint, record in persisted.items():
                container = self.containers[endpoint]
                container.state = ContainerState.QUARANTINED
                container.failures = int(record.get("failures", 0))
                container.last_error = str(
                    record.get("reason") or "persisted quarantine"
                )
        for endpoint in persisted:
            await self._schedule_recovery(endpoint)

    async def acquire(self, timeout: float = 300) -> dict[str, str]:
        deadline = time.monotonic() + timeout
        async with self.condition:
            while True:
                active_capacity = sum(
                    item.role == ContainerRole.ACTIVE for item in self.containers.values()
                )
                recovering = any(
                    item.state == ContainerState.RECOVERING
                    for item in self.containers.values()
                )
                serviceable = sum(
                    item.state in {ContainerState.AVAILABLE, ContainerState.LEASED}
                    for item in self.containers.values()
                )
                if serviceable < active_capacity and not recovering:
                    raise RuntimeError(
                        "healthy MobileWorld capacity is below the configured "
                        f"active count ({serviceable}/{active_capacity})"
                    )
                leased_total = sum(
                    item.state == ContainerState.LEASED for item in self.containers.values()
                )
                if serviceable >= active_capacity and leased_total < active_capacity:
                    for container in self.containers.values():
                        if (
                            container.role == ContainerRole.ACTIVE
                            and container.state == ContainerState.AVAILABLE
                        ):
                            return self._lease(container)

                unavailable_active = sum(
                    item.role == ContainerRole.ACTIVE
                    and item.state
                    in {
                        ContainerState.QUARANTINED,
                        ContainerState.RECOVERING,
                        ContainerState.RETIRED,
                    }
                    for item in self.containers.values()
                )
                leased_spares = sum(
                    item.role == ContainerRole.SPARE and item.state == ContainerState.LEASED
                    for item in self.containers.values()
                )
                if (
                    serviceable >= active_capacity
                    and leased_total < active_capacity
                    and leased_spares < unavailable_active
                ):
                    for container in self.containers.values():
                        if (
                            container.role == ContainerRole.SPARE
                            and container.state == ContainerState.AVAILABLE
                        ):
                            return self._lease(container)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a healthy MobileWorld container")
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=remaining)
                except TimeoutError as exc:
                    raise TimeoutError(
                        "timed out waiting for a healthy MobileWorld container"
                    ) from exc

    @staticmethod
    def _lease(container: AsyncContainer) -> dict[str, str]:
        container.state = ContainerState.LEASED
        container.token = uuid.uuid4().hex
        return {
            "endpoint": container.endpoint,
            "token": container.token,
            "role": container.role.value,
        }

    async def release(self, lease: dict[str, str]) -> None:
        async with self.condition:
            container = self._validate(lease)
            self._initializing.discard(container.token)
            self._tearing_down.discard(container.token)
            container.state = ContainerState.AVAILABLE
            container.token = None
            self.condition.notify_all()

    async def quarantine(self, lease: dict[str, str], reason: str) -> None:
        endpoint: str
        async with self.condition:
            container = self._validate(lease)
            self._initializing.discard(container.token)
            self._tearing_down.discard(container.token)
            container.state = ContainerState.QUARANTINED
            container.token = None
            container.failures += 1
            container.last_error = reason
            endpoint = container.endpoint
            self.condition.notify_all()
        if self.recovery is not None:
            await asyncio.to_thread(
                self.recovery.mark, endpoint, "quarantined", reason
            )
            await self._schedule_recovery(endpoint)

    async def begin_initialization(
        self, lease: dict[str, str], timeout: float = 300
    ) -> None:
        """Limit concurrent snapshot restores without limiting active rollouts."""
        deadline = time.monotonic() + timeout
        async with self.condition:
            container = self._validate(lease)
            token = container.token
            assert token is not None
            while len(self._initializing) >= self.initialization_concurrency:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out waiting for a MobileWorld initialization slot"
                    )
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=remaining)
                except TimeoutError as exc:
                    raise TimeoutError(
                        "timed out waiting for a MobileWorld initialization slot"
                    ) from exc
                container = self._validate(lease)
            self._initializing.add(token)

    async def end_initialization(self, lease: dict[str, str]) -> None:
        async with self.condition:
            container = self._validate(lease)
            token = container.token
            if token not in self._initializing:
                raise ValueError("lease does not own an initialization slot")
            self._initializing.remove(token)
            self.condition.notify_all()

    async def begin_teardown(
        self, lease: dict[str, str], timeout: float = 300
    ) -> None:
        """Limit nested Docker/OverlayFS teardown pressure without limiting rollouts."""
        deadline = time.monotonic() + timeout
        async with self.condition:
            container = self._validate(lease)
            token = container.token
            assert token is not None
            while len(self._tearing_down) >= self.teardown_concurrency:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out waiting for a MobileWorld teardown slot"
                    )
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=remaining)
                except TimeoutError as exc:
                    raise TimeoutError(
                        "timed out waiting for a MobileWorld teardown slot"
                    ) from exc
                container = self._validate(lease)
            self._tearing_down.add(token)

    async def end_teardown(self, lease: dict[str, str]) -> None:
        async with self.condition:
            container = self._validate(lease)
            token = container.token
            if token not in self._tearing_down:
                raise ValueError("lease does not own a teardown slot")
            self._tearing_down.remove(token)
            self.condition.notify_all()

    async def recover(self, endpoint: str) -> None:
        normalized = endpoint.rstrip("/")
        async with self.condition:
            container = self.containers[normalized]
            if container.state == ContainerState.LEASED:
                raise ValueError("cannot recover a leased container")
            container.state = ContainerState.AVAILABLE
            container.last_error = None
            self.condition.notify(1)
        if self.recovery is not None:
            await asyncio.to_thread(
                self.recovery.mark, normalized, "available", None
            )

    async def snapshot(self) -> list[dict[str, object]]:
        async with self.condition:
            return [
                {
                    "endpoint": item.endpoint,
                    "role": item.role.value,
                    "state": item.state.value,
                    "failures": item.failures,
                    "last_error": item.last_error,
                }
                for item in self.containers.values()
            ]

    async def _schedule_recovery(self, endpoint: str) -> None:
        if self.recovery is None:
            return
        async with self.condition:
            existing = self._recovery_tasks.get(endpoint)
            if existing is not None and not existing.done():
                return
            container = self.containers[endpoint]
            if container.state == ContainerState.LEASED:
                raise ValueError("cannot recover a leased container")
            container.state = ContainerState.RECOVERING
            task = asyncio.create_task(self._recover_endpoint(endpoint))
            self._recovery_tasks[endpoint] = task
            self.condition.notify_all()

    async def _recover_endpoint(self, endpoint: str) -> None:
        assert self.recovery is not None
        try:
            async with self._recovery_semaphore:
                await asyncio.to_thread(self.recovery.recover_endpoint, endpoint)
        except BaseException as exc:
            async with self.condition:
                container = self.containers[endpoint]
                if container.state == ContainerState.RECOVERING:
                    container.state = ContainerState.QUARANTINED
                    container.last_error = f"recovery failed: {exc}"
                    self.condition.notify_all()
            return
        async with self.condition:
            container = self.containers[endpoint]
            if container.state == ContainerState.RECOVERING:
                container.state = ContainerState.AVAILABLE
                container.last_error = None
                self.condition.notify_all()

    def _validate(self, lease: dict[str, str]) -> AsyncContainer:
        container = self.containers.get(lease.get("endpoint", ""))
        if container is None or container.state != ContainerState.LEASED:
            raise ValueError("stale or foreign lease")
        if container.token != lease.get("token"):
            raise ValueError("stale or foreign lease")
        return container
