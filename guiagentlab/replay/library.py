"""Access to pre-scored success trajectories used by replay injection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guiagentlab.training.tracking import sha256_file


class ScoredReplayLibrary:
    """Load the included successful trajectories and process rewards."""

    def __init__(
        self,
        library: str | Path,
        scores: str | Path,
    ) -> None:
        self.root = Path(library).expanduser().resolve()
        self.scores_path = Path(scores).expanduser().resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self._rows = {
            str(row["task_name"]): row for row in manifest["episodes"]
        }
        if not self._rows:
            raise ValueError("success replay library is empty")
        self._scores = json.loads(self.scores_path.read_text(encoding="utf-8"))
        if not self._scores.get("complete"):
            raise ValueError(f"replay scores are incomplete: {self.scores_path}")

    @property
    def tasks(self) -> set[str]:
        return set(self._rows)

    def load(self, task_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        row = self._rows.get(task_name)
        if row is None:
            raise KeyError(f"replay library has no success path for {task_name}")
        episode_path = self.root / "episodes" / f"{row['episode_id']}.json"
        if sha256_file(episode_path) != str(row["episode_sha256"]):
            raise ValueError(f"replay episode changed: {episode_path}")
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        scores = []
        for step_index in range(len(episode["steps"])):
            key = f"{episode['episode_id']}:{step_index}"
            score = self._scores["steps"].get(key)
            if not isinstance(score, dict) or "reward" not in score:
                raise ValueError(f"replay step has no process reward: {key}")
            scores.append(score)
        return episode, scores

    def image_bytes(self, episode: dict[str, Any]) -> list[bytes]:
        output: list[bytes] = []
        for descriptor in episode["artifacts"]["screenshots"]:
            path = self.root / str(descriptor["path"])
            payload = path.read_bytes()
            if sha256_file(path) != str(descriptor["sha256"]):
                raise ValueError(f"replay screenshot changed: {path}")
            output.append(payload)
        return output
