"""Strict aggregation for persisted MobileWorld evaluation episodes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from guiagentlab.data.datasets import read_task_specs
from guiagentlab.rollout.recording import utc_now


def _read_json_files(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]


def summarize(
    episode_dir: str | Path,
    dataset_path: str | Path,
    *,
    expected_task_count: int = 105,
    expected_samples: int,
    mode: str,
) -> dict[str, Any]:
    root = Path(episode_dir)
    task_specs = read_task_specs(dataset_path)
    expected_tasks = sorted(
        task_specs.loc[
            ~task_specs["requires_external_network"].astype(bool), "task_name"
        ].astype(str)
    )
    if (
        len(expected_tasks) != expected_task_count
        or len(set(expected_tasks)) != expected_task_count
    ):
        raise ValueError(
            f"dataset must contain exactly {expected_task_count} unique local tasks"
        )

    episodes = _read_json_files(root / "episodes")
    infrastructure = _read_json_files(root / "infrastructure")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        if episode.get("valid") is not True:
            episode_id = episode.get("episode_id")
            raise ValueError(f"invalid episode was written as a model result: {episode_id}")
        grouped[str(episode.get("task_name"))].append(episode)

    expected_set = set(expected_tasks)
    unexpected_tasks = sorted(set(grouped) - expected_set)
    per_task = []
    incomplete = []
    success_total = 0
    for task_name in expected_tasks:
        rows = grouped.get(task_name, [])
        successes = sum(float(row["final_score"]) > 0 for row in rows)
        success_total += successes
        if len(rows) != expected_samples:
            incomplete.append(
                {"task_name": task_name, "expected": expected_samples, "actual": len(rows)}
            )
        per_task.append(
            {
                "task_name": task_name,
                "trajectory_count": len(rows),
                "success_count": successes,
                "trajectory_success_rate": successes / len(rows) if rows else None,
                "pass": bool(successes),
                "episode_ids": [str(row["episode_id"]) for row in rows],
            }
        )

    failure_kinds = Counter(
        str(row.get("error", {}).get("kind", "unknown")) for row in infrastructure
    )
    expected_trajectories = len(expected_tasks) * expected_samples
    pass_count = sum(bool(row["pass"]) for row in per_task)
    complete = (
        not incomplete
        and not unexpected_tasks
        and len(episodes) == expected_trajectories
        and not list((root / "internal_errors").glob("*.json"))
    )
    metric_name = "pass_at_1" if expected_samples == 1 else f"pass_at_{expected_samples}"
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "mode": mode,
        "expected_tasks": len(expected_tasks),
        "expected_samples_per_task": expected_samples,
        "expected_trajectories": expected_trajectories,
        "valid_trajectories": len(episodes),
        "successful_trajectories": success_total,
        "trajectory_success_rate": success_total / len(episodes) if episodes else None,
        metric_name: pass_count / len(expected_tasks),
        "passed_tasks": pass_count,
        "infrastructure_attempts_discarded": len(infrastructure),
        "infrastructure_failures_by_kind": dict(sorted(failure_kinds.items())),
        "unexpected_tasks": unexpected_tasks,
        "incomplete_tasks": incomplete,
        "complete": complete,
        "per_task": per_task,
    }


def write_summary(summary: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--expected-tasks", default=105, type=int)
    parser.add_argument("--mode", required=True, choices=("greedy", "sample"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(
        args.episodes,
        args.dataset,
        expected_task_count=args.expected_tasks,
        expected_samples=args.expected_samples,
        mode=args.mode,
    )
    json_path = write_summary(result, args.output)
    print(json_path)
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
