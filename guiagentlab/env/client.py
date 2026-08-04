"""Small, strict client for the official MobileWorld HTTP API."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any

import requests

from guiagentlab.env.errors import FailureKind, InfrastructureError
from guiagentlab.reward.milestones import MilestoneSnapshot


@dataclass(frozen=True, slots=True)
class TaskInfo:
    name: str
    tags: tuple[str, ...] = ()
    apps: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Evaluation:
    score: float
    reason: str | None = None


class MobileWorldClient:
    """One client per leased container.

    Proxy environment variables are deliberately ignored: container endpoints are
    local infrastructure and must never be routed through an HTTP proxy.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        device: str = "emulator-5554",
        timeout: float = 30.0,
        step_wait_time: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        if step_wait_time < 0:
            raise ValueError("step_wait_time must be non-negative")
        self.endpoint = endpoint.rstrip("/")
        self.device = device
        self.timeout = timeout
        self.step_wait_time = step_wait_time
        self.session = session or requests.Session()
        self.session.trust_env = False

    def _request(
        self,
        method: str,
        path: str,
        *,
        kind: FailureKind,
        retryable: bool = True,
        **kwargs: Any,
    ) -> Any:
        url = f"{self.endpoint}{path}"
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.Timeout as exc:
            raise InfrastructureError(FailureKind.TIMEOUT, str(exc), self.endpoint) from exc
        except requests.RequestException as exc:
            raise InfrastructureError(kind, str(exc), self.endpoint, retryable) from exc

        if not 200 <= response.status_code < 300:
            try:
                body = response.json()
            except ValueError:
                body = response.text[:1000]
            raise InfrastructureError(
                kind,
                f"HTTP {response.status_code}",
                self.endpoint,
                retryable,
                {"body": body},
            )
        try:
            return response.json()
        except ValueError as exc:
            raise InfrastructureError(
                FailureKind.PROTOCOL,
                "response is not valid JSON",
                self.endpoint,
                retryable,
            ) from exc

    def health(self) -> dict[str, Any]:
        value = self._request("GET", "/health", kind=FailureKind.HEALTH)
        if (
            not isinstance(value, dict)
            or value.get("ok") is not True
            or value.get("ready", True) is not True
        ):
            raise InfrastructureError(
                FailureKind.HEALTH,
                "server reported unhealthy",
                self.endpoint,
                details={"response": value},
            )
        return value

    def initialize_controller(self) -> dict[str, Any]:
        value = self._request(
            "GET",
            "/init",
            kind=FailureKind.INITIALIZATION,
            params={"device": self.device},
        )
        if not isinstance(value, dict):
            raise InfrastructureError(
                FailureKind.PROTOCOL, "invalid controller response", self.endpoint
            )
        return value

    def tasks(self) -> list[TaskInfo]:
        value = self._request("GET", "/task/list", kind=FailureKind.PROTOCOL)
        if not isinstance(value, list):
            raise InfrastructureError(
                FailureKind.PROTOCOL, "task list is not a list", self.endpoint
            )
        return [
            TaskInfo(
                name=str(item["name"]),
                tags=tuple(str(tag) for tag in item.get("tags", [])),
                apps=tuple(str(app) for app in item.get("apps", [])),
            )
            for item in value
        ]

    def goal(self, task_name: str) -> str:
        value = self._request(
            "GET",
            "/task/goal",
            kind=FailureKind.PROTOCOL,
            params={"task_name": task_name},
        )
        if not isinstance(value, str):
            raise InfrastructureError(FailureKind.PROTOCOL, "task goal is not text", self.endpoint)
        return value

    def initialize_task(self, task_name: str) -> None:
        self._request(
            "POST",
            "/task/init",
            kind=FailureKind.INITIALIZATION,
            # Initialization changes emulator state and is not safe to replay blindly.
            retryable=False,
            json={"task_name": task_name, "req_device": self.device},
        )

    def screenshot_png(self, *, wait_to_stabilize: bool = True) -> bytes:
        # Match MobileWorld's AndroidEnvClient contract: the delay belongs to
        # observation capture, so both the initial frame and every post-action
        # frame represent a settled UI rather than an animation/loading state.
        if wait_to_stabilize and self.step_wait_time:
            time.sleep(self.step_wait_time)
        value = self._request(
            "GET",
            "/screenshot",
            kind=FailureKind.STEP,
            params={"device": self.device, "return_b64": "true"},
        )
        try:
            png = base64.b64decode(value["b64_png"], validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise InfrastructureError(
                FailureKind.PROTOCOL, "screenshot has no valid b64_png", self.endpoint
            ) from exc
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise InfrastructureError(
                FailureKind.PROTOCOL,
                "screenshot payload is empty or is not PNG",
                self.endpoint,
                details={"payload_bytes": len(png)},
            )
        return png

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        value = self._request(
            "POST",
            "/step",
            kind=FailureKind.STEP,
            # A click/text action might have executed before the connection failed.
            retryable=False,
            json={"device": self.device, "action": action},
        )
        if not isinstance(value, dict) or "result" not in value:
            raise InfrastructureError(FailureKind.PROTOCOL, "invalid step response", self.endpoint)
        return value

    def evaluate(self, task_name: str) -> Evaluation:
        value = self._request(
            "GET",
            "/task/eval",
            kind=FailureKind.EVALUATION,
            json={"task_name": task_name, "req_device": self.device},
        )
        try:
            score = float(value["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InfrastructureError(
                FailureKind.PROTOCOL, "evaluation has no numeric score", self.endpoint
            ) from exc
        return Evaluation(score=score, reason=value.get("reason"))

    def milestones(self, task_name: str) -> MilestoneSnapshot:
        """Read exact task progress from the environment backend."""
        value = self._request(
            "GET",
            "/task/milestones",
            kind=FailureKind.EVALUATION,
            json={"task_name": task_name, "req_device": self.device},
        )
        try:
            return MilestoneSnapshot.from_payload(value)
        except ValueError as exc:
            raise InfrastructureError(
                FailureKind.PROTOCOL,
                f"invalid milestone response: {exc}",
                self.endpoint,
            ) from exc

    def tear_down(self, task_name: str) -> None:
        self._request(
            "POST",
            "/task/tear_down",
            kind=FailureKind.TEARDOWN,
            retryable=False,
            json={"task_name": task_name, "req_device": self.device},
        )
