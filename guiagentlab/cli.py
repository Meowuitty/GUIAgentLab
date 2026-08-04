"""GUIAgentLab command line interface."""

from __future__ import annotations

import argparse
import json

from guiagentlab.training.launch import PUBLIC_METHODS, launch_training
from guiagentlab.validation import preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gal")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="launch one supported training method")
    train.add_argument("method", choices=PUBLIC_METHODS)
    train.add_argument("--model")
    train.add_argument("--teacher-model")
    train.add_argument("--train-file")
    train.add_argument("--validation-file")
    train.add_argument("--replay", choices=("none", "success"), default="none")
    train.add_argument("--set-env", action="append", default=[])
    train.add_argument("--dry-run", action="store_true")

    validate = commands.add_parser("validate", help="run read-only environment preflight")
    validate.add_argument("--servers", required=True)
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--active", type=int, default=32)
    validate.add_argument("--spares", type=int, default=8)
    validate.add_argument(
        "--recover",
        action="store_true",
        help="restart and strongly probe failed outer containers before validation",
    )
    validate.add_argument(
        "--warm",
        action="store_true",
        help="load one task snapshot on every endpoint before validation",
    )
    validate.add_argument("--warm-concurrency", type=int, default=8)

    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train":
        result = launch_training(
            args.method,
            model=args.model,
            teacher_model=args.teacher_model,
            train_file=args.train_file,
            validation_file=args.validation_file,
            replay=args.replay,
            environment=args.set_env,
            dry_run=args.dry_run,
        )
    elif args.command == "validate":
        result = preflight(
            args.servers,
            args.dataset,
            active=args.active,
            spares=args.spares,
            recover=args.recover,
            warm=args.warm,
            warm_concurrency=args.warm_concurrency,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
