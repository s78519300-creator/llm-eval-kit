"""Command-line interface: ``llm-eval run | checks | validate``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from . import __version__
from .checks import available_checks
from .dataset import DatasetError, load_jsonl
from .evaluator import Evaluator
from .report import render_table, write_reports
from .rubric import RubricError, load_rubric

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _formats(value: str) -> list[str]:
    formats = [f.strip().lower() for f in value.split(",") if f.strip()]
    bad = [f for f in formats if f not in ("md", "json")]
    if bad or not formats:
        raise argparse.ArgumentTypeError("formats must be a comma-separated list of md,json")
    return formats


def _unit_float(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Rubric-based, deterministic evaluation of LLM outputs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    run = sub.add_parser("run", help="evaluate a JSONL dataset against a rubric")
    run.add_argument("--dataset", "-d", required=True, type=Path, help="JSONL file of samples")
    run.add_argument("--rubric", "-r", required=True, type=Path, help="rubric (.json/.yaml)")
    run.add_argument("--out", "-o", type=Path, help="directory for report.md / report.json")
    run.add_argument(
        "--format",
        dest="formats",
        type=_formats,
        default=["md", "json"],
        help="report formats to write (default: md,json)",
    )
    run.add_argument(
        "--pass-threshold", type=_unit_float, help="override the rubric's pass threshold"
    )
    run.add_argument(
        "--fail-under",
        type=_unit_float,
        metavar="RATE",
        help="exit with status 1 if the pass rate is below RATE (0-1)",
    )
    run.add_argument("--strict", action="store_true", help="exit with status 1 if any sample fails")
    run.add_argument("--quiet", "-q", action="store_true", help="do not print the summary table")

    sub.add_parser("checks", help="list the built-in checks")

    validate = sub.add_parser("validate", help="validate a rubric file")
    validate.add_argument("rubric", type=Path)
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    rubric = load_rubric(args.rubric)
    if args.pass_threshold is not None:
        rubric.pass_threshold = args.pass_threshold
    samples = load_jsonl(args.dataset)
    if not samples:
        raise DatasetError(f"dataset is empty: {args.dataset}")

    run = Evaluator(rubric).run(samples, dataset=args.dataset.as_posix())

    if not args.quiet:
        print(render_table(run))
    if args.out:
        paths = write_reports(run, args.out, args.formats)
        if not args.quiet:
            print()
            for path in paths:
                print(f"Wrote {path.as_posix()}")

    if args.strict and run.passed < len(run.results):
        return EXIT_FAILED
    if args.fail_under is not None and run.pass_rate + 1e-9 < args.fail_under:
        return EXIT_FAILED
    return EXIT_OK


def _cmd_checks(_: argparse.Namespace) -> int:
    checks = available_checks()
    width = max(len(name) for name in checks)
    for name, summary in checks.items():
        print(f"{name.ljust(width)}  {summary}")
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    rubric = load_rubric(args.rubric)
    Evaluator(rubric)  # resolves checks and validates their params
    print(
        f"OK: {rubric.name} v{rubric.version} - {len(rubric.criteria)} criteria, "
        f"total weight {rubric.total_weight:g}, pass threshold {rubric.pass_threshold:.2f}"
    )
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``llm-eval`` console script."""
    args = build_parser().parse_args(argv)
    handlers = {"run": _cmd_run, "checks": _cmd_checks, "validate": _cmd_validate}
    try:
        return handlers[args.command](args)
    except (RubricError, DatasetError) as exc:
        print(f"llm-eval: error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
