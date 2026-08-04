"""Safe retention helpers for resumable VERL checkpoints."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_CHECKPOINT_DIRECTORY = re.compile(r"global_step_(\d+)")


def prune_completed_checkpoints(
    checkpoint_root: str | Path,
    *,
    fixed_frequency: int,
) -> list[Path]:
    """Keep the latest complete checkpoint and fixed-frequency milestones.

    VERL writes ``latest_checkpointed_iteration.txt`` only after the actor and
    dataloader checkpoint have completed.  Treat that marker as the deletion
    boundary so an interrupted or partially written checkpoint is never used
    as evidence that an older recovery point can be removed.
    """

    root = Path(checkpoint_root)
    if fixed_frequency <= 0:
        raise ValueError("fixed checkpoint frequency must be positive")

    latest_marker = root / "latest_checkpointed_iteration.txt"
    if not latest_marker.is_file():
        return []
    try:
        latest_step = int(latest_marker.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise ValueError(f"invalid checkpoint marker: {latest_marker}") from error

    latest_directory = root / f"global_step_{latest_step}"
    if not latest_directory.is_dir() or not (latest_directory / "data.pt").is_file():
        return []

    removed: list[Path] = []
    for candidate in root.iterdir():
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        match = _CHECKPOINT_DIRECTORY.fullmatch(candidate.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step >= latest_step or step % fixed_frequency == 0:
            continue
        shutil.rmtree(candidate)
        removed.append(candidate)
    return removed
