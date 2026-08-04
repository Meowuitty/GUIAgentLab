"""Host-owned recovery for disposable outer MobileWorld containers.

This module deliberately does not modify or coordinate MobileWorld's internal
emulator restart logic. A failed endpoint is removed from scheduling first; the
outer container is then restarted so its GUIAgentLab entrypoint can sanitize
the per-slot mutable state while preserving the nested-image and AVD caches.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from guiagentlab.env.client import MobileWorldClient

_SAFE_CONTAINER_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    state_path: str
    container_prefix: str = "guiagentlab_mw"
    start_port: int = 20000
    ready_timeout: float = 1200.0
    request_timeout: float = 30.0
    probe_task: str = "AdjustBrightnessMaximumTask"
    attempts: int = 3
    retry_backoff: float = 60.0

    def __post_init__(self) -> None:
        if not _SAFE_CONTAINER_PREFIX.fullmatch(self.container_prefix):
            raise ValueError(f"unsafe MobileWorld container prefix: {self.container_prefix!r}")
        if not 1 <= self.start_port <= 65535:
            raise ValueError("MobileWorld start port is out of range")
        if self.ready_timeout <= 0 or self.request_timeout <= 0:
            raise ValueError("recovery timeouts must be positive")
        if self.attempts <= 0 or self.retry_backoff < 0:
            raise ValueError("recovery attempts/backoff are invalid")
        if not self.probe_task:
            raise ValueError("recovery probe task must not be empty")

    @classmethod
    def from_environment(cls) -> RecoveryConfig | None:
        if os.environ.get("GUIAGENTLAB_ENDPOINT_RECOVERY", "1") == "0":
            return None
        state_root = os.environ.get("GUIAGENTLAB_STATE_DIR")
        explicit_path = os.environ.get("GUIAGENTLAB_ENDPOINT_STATE")
        if not explicit_path and not state_root:
            return None
        state_path = explicit_path or str(
            Path(state_root).resolve()
            / "mobileworld"
            / "runtime"
            / "endpoint-state.json"
        )
        return cls(
            state_path=state_path,
            container_prefix=os.environ.get(
                "GUIAGENTLAB_MW_PREFIX", "guiagentlab_mw"
            ),
            start_port=int(os.environ.get("GUIAGENTLAB_MW_START_PORT", "20000")),
            ready_timeout=float(
                os.environ.get("GUIAGENTLAB_ENDPOINT_RECOVERY_TIMEOUT", "1200")
            ),
            request_timeout=float(
                os.environ.get("GUIAGENTLAB_ENDPOINT_PROBE_TIMEOUT", "180")
            ),
            probe_task=os.environ.get(
                "GUIAGENTLAB_ENDPOINT_PROBE_TASK",
                "AdjustBrightnessMaximumTask",
            ),
            attempts=int(
                os.environ.get("GUIAGENTLAB_ENDPOINT_RECOVERY_ATTEMPTS", "3")
            ),
            retry_backoff=float(
                os.environ.get("GUIAGENTLAB_ENDPOINT_RECOVERY_BACKOFF", "60")
            ),
        )

    def serializable(self) -> dict[str, Any]:
        return asdict(self)


class EndpointRecovery:
    """Persist endpoint disposition and restart one exact outer container."""

    def __init__(self, config: RecoveryConfig) -> None:
        self.config = config
        self.state_path = Path(config.state_path).resolve()
        self.lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")

    @classmethod
    def from_serializable(cls, value: dict[str, Any] | None) -> EndpointRecovery | None:
        if value is None:
            return None
        return cls(RecoveryConfig(**value))

    @contextmanager
    def _locked_state(self) -> Iterator[dict[str, Any]]:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if self.state_path.is_file():
                    payload = json.loads(self.state_path.read_text(encoding="utf-8"))
                else:
                    payload = {
                        "schema_version": 1,
                        "updated_at": _utc_now(),
                        "endpoints": {},
                    }
                if payload.get("schema_version") != 1 or not isinstance(
                    payload.get("endpoints"), dict
                ):
                    raise ValueError(f"invalid endpoint state file: {self.state_path}")
                yield payload
                payload["updated_at"] = _utc_now()
                fd, temporary_name = tempfile.mkstemp(
                    dir=self.state_path.parent,
                    prefix=f".{self.state_path.name}.",
                    suffix=".tmp",
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as temporary:
                        json.dump(payload, temporary, ensure_ascii=False, indent=2)
                        temporary.write("\n")
                        temporary.flush()
                        os.fsync(temporary.fileno())
                    os.replace(temporary_name, self.state_path)
                finally:
                    if os.path.exists(temporary_name):
                        os.unlink(temporary_name)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def statuses(self, endpoints: list[str]) -> dict[str, dict[str, Any]]:
        normalized = {endpoint.rstrip("/") for endpoint in endpoints}
        with self._locked_state() as payload:
            records = payload["endpoints"]
            return {
                endpoint: dict(records[endpoint])
                for endpoint in normalized
                if endpoint in records
                and records[endpoint].get("state") != "available"
            }

    def mark(self, endpoint: str, state: str, reason: str | None = None) -> None:
        if state not in {"available", "quarantined", "recovering", "retired"}:
            raise ValueError(f"unsupported persistent endpoint state: {state}")
        endpoint = endpoint.rstrip("/")
        with self._locked_state() as payload:
            records = payload["endpoints"]
            previous = records.get(endpoint, {})
            failures = int(previous.get("failures", 0))
            if state == "quarantined" and previous.get("state") != "quarantined":
                failures += 1
            records[endpoint] = {
                "state": state,
                "reason": reason,
                "failures": failures,
                "updated_at": _utc_now(),
            }

    def _container_identity(self, endpoint: str) -> tuple[str, int]:
        parsed = urlparse(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError(f"recovery only accepts local HTTP endpoints: {endpoint!r}")
        if parsed.port is None:
            raise ValueError(f"endpoint has no port: {endpoint!r}")
        index = parsed.port - self.config.start_port
        if not 0 <= index < 4096:
            raise ValueError(
                f"endpoint port does not belong to the configured pool: {endpoint!r}"
            )
        return f"{self.config.container_prefix}_{index}", parsed.port

    @staticmethod
    def _run_docker(*args: str, timeout: float = 60.0) -> str:
        result = subprocess.run(
            ["docker", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()

    def _verify_container_mapping(self, name: str, endpoint_port: int) -> None:
        raw = self._run_docker("inspect", name)
        inspected = json.loads(raw)
        if not isinstance(inspected, list) or len(inspected) != 1:
            raise RuntimeError(f"docker inspect returned no unique container for {name}")
        record = inspected[0]
        # Docker clears NetworkSettings.Ports after a container exits.  The
        # immutable HostConfig mapping remains available and is the authority
        # needed before restarting an exited container.
        bindings = (
            record.get("NetworkSettings", {}).get("Ports", {}).get("6800/tcp")
            or record.get("HostConfig", {}).get("PortBindings", {}).get("6800/tcp")
        )
        host_ports = {
            int(item["HostPort"])
            for item in bindings or []
            if isinstance(item, dict) and str(item.get("HostPort", "")).isdigit()
        }
        if endpoint_port not in host_ports:
            raise RuntimeError(
                f"container {name} is not bound to MobileWorld port {endpoint_port}"
            )

    def _wait_ready(self, name: str, endpoint: str) -> None:
        deadline = time.monotonic() + self.config.ready_timeout
        last_error = "container did not become healthy"
        while time.monotonic() < deadline:
            try:
                status = self._run_docker(
                    "inspect",
                    "--format",
                    "{{.State.Status}} "
                    "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
                    name,
                    timeout=15,
                )
                if status == "running healthy":
                    MobileWorldClient(
                        endpoint, timeout=self.config.request_timeout, step_wait_time=0
                    ).health()
                    return
                last_error = f"container state is {status!r}"
            except BaseException as exc:
                last_error = str(exc)
            time.sleep(2)
        raise TimeoutError(
            f"outer MobileWorld container {name} failed readiness: {last_error}"
        )

    def _accept_recovered_endpoint(self, endpoint: str) -> None:
        client = MobileWorldClient(
            endpoint,
            timeout=self.config.request_timeout,
            step_wait_time=0,
        )
        initialized = False
        client.health()
        client.initialize_controller()
        # Verify the freshly started emulator before exercising snapshot restore.
        client.screenshot_png(wait_to_stabilize=False)
        try:
            client.initialize_task(self.config.probe_task)
            initialized = True
            # Task initialization loads init_state; a second screenshot proves
            # ADB and qemu survived that restore.
            client.screenshot_png(wait_to_stabilize=False)
        finally:
            if initialized:
                client.tear_down(self.config.probe_task)
        client.health()

    def recover_endpoint(self, endpoint: str) -> None:
        endpoint = endpoint.rstrip("/")
        name, endpoint_port = self._container_identity(endpoint)
        last_error: Exception | None = None
        for attempt in range(1, self.config.attempts + 1):
            self.mark(
                endpoint,
                "recovering",
                f"outer container restart attempt {attempt}/{self.config.attempts}",
            )
            try:
                self._verify_container_mapping(name, endpoint_port)
                restart_error: Exception | None = None
                try:
                    self._run_docker(
                        "restart",
                        "--time",
                        "30",
                        name,
                        timeout=min(self.config.ready_timeout, 180.0),
                    )
                except Exception as exc:
                    # Docker may report a timeout/non-zero exit after the
                    # container has already reached a usable state.  Treat the
                    # endpoint probe, not the CLI exit code, as authoritative.
                    restart_error = exc
                try:
                    self._wait_ready(name, endpoint)
                    self._accept_recovered_endpoint(endpoint)
                except Exception as exc:
                    if restart_error is not None:
                        raise RuntimeError(
                            "outer restart command failed and the endpoint did "
                            f"not recover: restart={restart_error}; probe={exc}"
                        ) from exc
                    raise
            except Exception as exc:
                last_error = exc
                if attempt == self.config.attempts:
                    self.mark(endpoint, "quarantined", f"recovery failed: {exc}")
                    raise
                delay = self.config.retry_backoff * attempt
                self.mark(
                    endpoint,
                    "recovering",
                    f"recovery attempt {attempt} failed; retrying in {delay:g}s: {exc}",
                )
                time.sleep(delay)
            else:
                self.mark(endpoint, "available", None)
                return
        assert last_error is not None
        raise last_error
