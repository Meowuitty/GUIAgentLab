"""Canonical method-oriented training launcher."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from guiagentlab.config import PROJECT_ROOT

PUBLIC_METHODS = ("grpo", "gigpo", "admire", "opd", "opd-teacher")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _method_config(method: str) -> dict[str, Any]:
    if method not in PUBLIC_METHODS:
        raise ValueError(f"unsupported training method: {method}")
    filename = method.replace("-", "_") + ".yaml"
    base = yaml.safe_load(
        (PROJECT_ROOT / "configs/train/base.yaml").read_text(encoding="utf-8")
    )
    specific = yaml.safe_load(
        (PROJECT_ROOT / "configs/train" / filename).read_text(encoding="utf-8")
    )
    return {
        **base,
        **specific,
        "environment": {
            **(base.get("environment") or {}),
            **(specific.get("environment") or {}),
        },
    }


def _parse_environment(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, separator, content = value.partition("=")
        if not separator or not _ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError(f"environment override must be NAME=VALUE: {value!r}")
        parsed[name] = content
    return parsed


def launch_training(
    method: str,
    *,
    model: str | None = None,
    teacher_model: str | None = None,
    train_file: str | None = None,
    validation_file: str | None = None,
    replay: str = "none",
    environment: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resolve one public method and either report or replace the process."""
    config = _method_config(method)
    allowed_replay = tuple(config.get("replay") or ("none",))
    if replay not in allowed_replay:
        raise ValueError(
            f"{method} supports replay modes {allowed_replay}, not {replay!r}"
        )

    script = PROJECT_ROOT / str(config["script"])
    if not script.is_file():
        raise FileNotFoundError(f"training implementation is missing: {script}")
    child_environment = os.environ.copy()
    child_environment.update(
        {
            str(name): str(value)
            for name, value in (config.get("environment") or {}).items()
        }
    )
    child_environment.update(_parse_environment(environment or []))
    child_environment["SUCCESS_REPLAY_ENABLED"] = (
        "true" if replay == "success" else "false"
    )
    for name, value in (
        ("MODEL_PATH", model),
        ("TEACHER_MODEL_PATH", teacher_model),
        ("TRAIN_FILE", train_file),
        ("VALIDATION_FILE", validation_file),
    ):
        if value is not None:
            child_environment[name] = str(Path(value).expanduser())

    command = [str(script)]
    public = {
        "method": method,
        "algorithm": str(config["algorithm"]),
        "command": command,
        "replay": replay,
        "required_environment": list(config.get("required_environment") or []),
        "configured_environment": sorted(
            name
            for name in (
                "MODEL_PATH",
                "TEACHER_MODEL_PATH",
                "TRAIN_FILE",
                "VALIDATION_FILE",
                "GUIAGENTLAB_STATE_DIR",
                "GUIAGENTLAB_SERVER_FILE",
            )
            if child_environment.get(name)
        ),
    }
    if dry_run:
        return public

    os.execvpe(command[0], command, child_environment)
    raise AssertionError("os.execvpe returned unexpectedly")
