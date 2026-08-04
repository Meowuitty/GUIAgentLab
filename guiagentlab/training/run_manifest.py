"""Run-manifest writer used by the shell training entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from guiagentlab.config import resolve_project_path
from guiagentlab.training.tracking import build_run_manifest, write_json_atomic


def prepare_run(
    *,
    algorithm: str,
    run_id: str,
    output_dir: str | Path,
    entrypoint: str | Path,
    data_files: list[str | Path],
    command: list[str],
    wandb_enabled: bool,
) -> Path:
    run_dir = resolve_project_path(output_dir)
    manifest = build_run_manifest(
        algorithm=algorithm,
        command=command,
        data_files=data_files,
        entrypoint=entrypoint,
        run_id=run_id,
        wandb_enabled=wandb_enabled,
    )
    manifest_path = run_dir / run_id / "run-manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a training run manifest")
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=("grpo", "admire-grpo", "gigpo", "opd"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--data-file", action="append", default=[])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = _parser().parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("the training command is required after --")
    path = prepare_run(
        algorithm=args.algorithm,
        run_id=args.run_id,
        output_dir=args.output_dir,
        entrypoint=args.entrypoint,
        data_files=args.data_file,
        command=command,
        wandb_enabled=args.wandb,
    )
    print(path)


if __name__ == "__main__":
    main()
