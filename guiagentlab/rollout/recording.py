"""Compact, durable records for MobileWorld trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def json_value(value: Any) -> Any:
    """Convert common numpy/path/container values into JSON-safe objects."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if hasattr(value, "item"):
        try:
            return json_value(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


class EpisodeRecorder:
    """Write one authoritative episode plus lightweight task-name links."""

    schema_version = 2

    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root).resolve() if root else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def new_episode_id(self) -> str:
        return uuid4().hex

    def write_screenshot(
        self,
        episode_id: str,
        attempt: int,
        sequence: int,
        png: bytes,
    ) -> dict[str, Any] | None:
        if self.root is None:
            return None
        relative = (
            Path("screenshots")
            / episode_id
            / f"attempt-{attempt:02d}"
            / f"{sequence:03d}.png"
        )
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_bytes(destination, png)
        return {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(png).hexdigest(),
            "bytes": len(png),
        }

    def write_episode(self, episode_id: str, payload: dict[str, Any]) -> Path | None:
        if self.root is None:
            return None
        normalized = self._normalize_episode(payload)
        destination = self._write_json("episodes", f"{episode_id}.json", normalized)
        if destination is not None and normalized.get("task_name"):
            self._write_task_index(normalized, destination)
        return destination

    def write_infrastructure_attempt(
        self, episode_id: str, attempt: int, payload: dict[str, Any]
    ) -> Path | None:
        destination = self._write_json(
            "infrastructure",
            f"{episode_id}-attempt-{attempt:02d}.json",
            self._compact_failure(payload),
        )
        self._remove_attempt_screenshots(episode_id, attempt)
        return destination

    def write_internal_error(
        self, episode_id: str, attempt: int, payload: dict[str, Any]
    ) -> Path | None:
        destination = self._write_json(
            "internal_errors",
            f"{episode_id}-attempt-{attempt:02d}.json",
            self._compact_failure(payload),
        )
        self._remove_attempt_screenshots(episode_id, attempt)
        return destination

    @classmethod
    def _normalize_episode(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Store each screenshot descriptor once and reference it by index."""
        normalized = json_value(payload)
        screenshots: list[dict[str, Any]] = []
        positions: dict[str, int] = {}

        def register(value: Any) -> Any:
            if not isinstance(value, dict) or not value.get("path"):
                return value
            path = str(value["path"])
            if path not in positions:
                positions[path] = len(screenshots)
                screenshots.append(value)
            return positions[path]

        if "initial_screenshot" in normalized:
            normalized["initial_screenshot"] = register(
                normalized["initial_screenshot"]
            )
        for step in normalized.get("steps") or []:
            for key in ("screenshot_before", "screenshot_after"):
                if key in step:
                    step[key] = register(step[key])
        if screenshots:
            normalized["artifacts"] = {"screenshots": screenshots}
        return normalized

    @staticmethod
    def _compact_failure(payload: dict[str, Any]) -> dict[str, Any]:
        """Keep failure attribution, not discarded trajectory tensors/images."""
        value = json_value(payload)
        fields = (
            "episode_id",
            "attempt",
            "task_name",
            "task_index",
            "sample_index",
            "model_endpoint",
            "started_at",
            "finished_at",
            "stage",
            "termination",
            "error",
            "cleanup_error",
            "container_disposition",
        )
        compact = {key: value[key] for key in fields if value.get(key) is not None}
        endpoint = value.get("environment_endpoint") or value.get("endpoint")
        if endpoint is not None:
            compact["endpoint"] = endpoint
        if value.get("container_role") is not None:
            compact["container_role"] = value["container_role"]
        return compact

    def _write_task_index(
        self, payload: dict[str, Any], episode_path: Path
    ) -> Path | None:
        assert self.root is not None
        task_name = self._safe_component(str(payload["task_name"]))
        episode_id = str(payload["episode_id"])
        if payload.get("evaluation_mode") in {"greedy", "sample"}:
            sample_index = int(payload.get("sample_index", 0) or 0)
            leaf = f"sample-{sample_index:02d}"
        else:
            leaf = f"episode-{episode_id[:12]}"
        sample_dir = self.root / "tasks" / task_name / leaf
        sample_dir.mkdir(parents=True, exist_ok=True)

        self._atomic_symlink(
            os.path.relpath(episode_path, sample_dir),
            sample_dir / "episode.json",
        )
        attempt = int(payload.get("successful_attempt", 1) or 1)
        screenshot_dir = (
            self.root
            / "screenshots"
            / episode_id
            / f"attempt-{attempt:02d}"
        )
        if screenshot_dir.is_dir():
            self._atomic_symlink(
                os.path.relpath(screenshot_dir, sample_dir),
                sample_dir / "screenshots",
                target_is_directory=True,
            )
        return sample_dir

    def _remove_attempt_screenshots(self, episode_id: str, attempt: int) -> None:
        if self.root is None:
            return
        attempt_dir = (
            self.root
            / "screenshots"
            / episode_id
            / f"attempt-{attempt:02d}"
        )
        shutil.rmtree(attempt_dir, ignore_errors=True)
        episode_dir = attempt_dir.parent
        try:
            episode_dir.rmdir()
        except OSError:
            pass

    def _write_json(
        self, category: str, name: str, payload: dict[str, Any]
    ) -> Path | None:
        if self.root is None:
            return None
        destination = self.root / category / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        envelope = {"schema_version": self.schema_version, **json_value(payload)}
        self._atomic_write_json_file(destination, envelope)
        return destination

    @staticmethod
    def _safe_component(value: str) -> str:
        component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return component or "unknown-task"

    @classmethod
    def _atomic_write_json_file(cls, destination: Path, payload: Any) -> None:
        content = (
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        cls._atomic_write_bytes(destination, content)

    @staticmethod
    def _atomic_symlink(
        target: str, destination: Path, *, target_is_directory: bool = False
    ) -> None:
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        try:
            temporary.symlink_to(target, target_is_directory=target_is_directory)
            if destination.is_symlink() or destination.is_file():
                destination.unlink()
            elif destination.is_dir():
                shutil.rmtree(destination)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_write_bytes(destination: Path, content: bytes) -> None:
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
