"""ADMIRE milestone traces for offline successful replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guiagentlab.reward import MilestoneSnapshot, MilestoneTracker
from guiagentlab.training.tracking import sha256_file


class MilestoneReplayLibrary:
    """Load successful trajectories with precomputed milestone traces."""

    def __init__(
        self,
        library: str | Path,
        milestones: str | Path,
    ) -> None:
        self.root = Path(library).expanduser().resolve()
        self.milestones_path = Path(milestones).expanduser().resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self._rows = {
            str(row["task_name"]): row for row in manifest["episodes"]
        }
        if not self._rows:
            raise ValueError("success replay library is empty")

        cache = json.loads(self.milestones_path.read_text(encoding="utf-8"))
        results = cache.get("results")
        if not isinstance(results, list):
            raise ValueError("ADMIRE milestone cache has no results list")
        self._traces = {
            str(result.get("task_name")): result
            for result in results
            if isinstance(result, dict) and result.get("task_name")
        }
        if len(self._traces) != len(results):
            raise ValueError("ADMIRE milestone cache has duplicate or malformed tasks")
        cached_tasks = set(self._traces)
        expected_tasks = set(self._rows)
        if cached_tasks != expected_tasks:
            raise ValueError(
                "ADMIRE milestone cache task set differs from eligible replay set: "
                f"missing={sorted(expected_tasks - cached_tasks)}, "
                f"unexpected={sorted(cached_tasks - expected_tasks)}"
            )
        for task_name in sorted(expected_tasks):
            self._validate_trace(task_name, self._rows[task_name], self._traces[task_name])

    def _load_episode(
        self,
        task_name: str,
        row: dict[str, Any],
    ) -> dict[str, Any]:
        episode_path = self.root / "episodes" / f"{row['episode_id']}.json"
        if sha256_file(episode_path) != str(row["episode_sha256"]):
            raise ValueError(f"replay episode changed after milestone caching: {episode_path}")
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        if str(episode.get("task_name")) != task_name:
            raise ValueError(f"replay episode task differs for {task_name}")
        return episode

    def _validate_trace(
        self,
        task_name: str,
        row: dict[str, Any],
        trace: dict[str, Any],
    ) -> None:
        episode = self._load_episode(task_name, row)
        if str(trace.get("reference_episode")) != str(episode["episode_id"]):
            raise ValueError(f"ADMIRE milestone cache episode differs for {task_name}")
        initial_states = trace.get("initial_states")
        step_milestones = trace.get("step_milestones")
        if not isinstance(initial_states, dict) or not isinstance(step_milestones, list):
            raise ValueError(f"ADMIRE milestone cache lacks step traces for {task_name}")
        if len(step_milestones) != len(episode["steps"]):
            raise ValueError(f"ADMIRE milestone step count differs for {task_name}")
        tracker = MilestoneTracker(
            MilestoneSnapshot(registered=True, states=initial_states)
        )
        for position, (episode_step, cached_step) in enumerate(
            zip(episode["steps"], step_milestones, strict=True)
        ):
            expected_index = int(episode_step.get("index", position))
            if int(cached_step.get("index", -1)) != expected_index:
                raise ValueError(
                    f"ADMIRE milestone cache index differs for {task_name} step {position}"
                )
            snapshot = MilestoneSnapshot(
                registered=True,
                states=cached_step.get("states"),
            )
            observation = tracker.observe(snapshot)
            expected = {
                "completed": list(observation.completed),
                "newly_completed": list(observation.newly_completed),
                "progress": float(observation.progress),
                "hit": float(observation.hit),
            }
            actual = {
                "completed": cached_step.get("completed"),
                "newly_completed": cached_step.get("newly_completed"),
                "progress": float(cached_step.get("progress", -1)),
                "hit": float(cached_step.get("hit", -1)),
            }
            if actual != expected:
                raise ValueError(
                    f"ADMIRE milestone cache observation differs for "
                    f"{task_name} step {position}"
                )

    @property
    def tasks(self) -> set[str]:
        return set(self._rows)

    def load(
        self,
        task_name: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        row = self._rows.get(task_name)
        if row is None:
            raise KeyError(f"replay library has no success path for {task_name}")
        episode = self._load_episode(task_name, row)
        trace = self._traces[task_name]
        return episode, [dict(step) for step in trace["step_milestones"]]

    def image_bytes(self, episode: dict[str, Any]) -> list[bytes]:
        output: list[bytes] = []
        for descriptor in episode["artifacts"]["screenshots"]:
            path = self.root / str(descriptor["path"])
            payload = path.read_bytes()
            if sha256_file(path) != str(descriptor["sha256"]):
                raise ValueError(f"replay screenshot changed: {path}")
            output.append(payload)
        return output
