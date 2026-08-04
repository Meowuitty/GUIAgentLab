"""Local source-of-truth metadata; W&B is a secondary sink."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from guiagentlab.config import PROJECT_ROOT, resolve_project_path
from guiagentlab.reward import ProcessRewardConfig


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Hash a source tree while ignoring interpreter/tool caches."""
    root = Path(path)
    ignored_parts = {".git", ".pytest_cache", ".ruff_cache", ".vscode", "__pycache__"}
    digest = hashlib.sha256()
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root)
        if not candidate.is_file() or ignored_parts.intersection(relative.parts):
            continue
        if candidate.suffix in {".pyc", ".pyo"}:
            continue
        relative_bytes = relative.as_posix().encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return "unborn"


def _repository_metadata() -> dict[str, Any]:
    return {
        "revision": git_revision(PROJECT_ROOT),
        "source_state": {
            "guiagentlab_tree_sha256": sha256_tree(PROJECT_ROOT / "guiagentlab"),
            "mobileworld_src_tree_sha256": sha256_tree(PROJECT_ROOT / "mobile_world" / "src"),
            "verl_tree_sha256": sha256_tree(PROJECT_ROOT / "verl"),
        },
    }


def build_run_manifest(
    *,
    algorithm: str,
    command: list[str],
    data_files: list[str | Path],
    entrypoint: str | Path,
    run_id: str,
    wandb_enabled: bool,
) -> dict[str, Any]:
    upstream = yaml.safe_load((PROJECT_ROOT / "configs/upstream.yaml").read_text(encoding="utf-8"))
    resolved_data_files = [resolve_project_path(path) for path in data_files]
    entrypoint_path = resolve_project_path(entrypoint)
    project_file = PROJECT_ROOT / "pyproject.toml"
    process_reward = (
        ProcessRewardConfig.from_environment().public_metadata() if algorithm == "gigpo" else None
    )
    return {
        "schema_version": 2,
        "run_id": run_id,
        "algorithm": algorithm,
        "created_at": datetime.now(UTC).isoformat(),
        "command": command,
        "configuration": {
            "source": "shell",
            "entrypoint": str(entrypoint_path),
            "sha256": sha256_file(entrypoint_path),
        },
        "repository": _repository_metadata(),
        "upstream": upstream,
        "environment": {
            "python": platform.python_version(),
            "project_file": str(project_file.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(project_file),
        },
        "datasets": {
            str(path): sha256_file(path) if path.is_file() else "missing"
            for path in resolved_data_files
        },
        "tracking": {"wandb_enabled": wandb_enabled},
        "process_reward": process_reward,
    }


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
