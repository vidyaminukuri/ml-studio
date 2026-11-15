"""Command line entry: ``python -m eda``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eda import run_analysis
from eda.analyze import AUTO_MAX_DISTINCT, EDAError
from eda.sample import CLASSIFICATION_CSV, CLASSIFICATION_TARGET, REGRESSION_CSV, REGRESSION_TARGET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis for a CSV with one target column.",
        epilog=(
            "With no --data and no --target, the built-in classification sample is used. "
            f"Auto task detection: a string, boolean, or categorical target is classification; "
            f"a numeric target is classification when it has at most {AUTO_MAX_DISTINCT} distinct "
            f"non-null values, and regression otherwise. "
            f"Regression sample: python -m eda --data {REGRESSION_CSV} --target {REGRESSION_TARGET} --out eda_report"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="CSV path. Omit, together with --target, to use the built-in classification sample.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column name. Required when --data is set.",
    )
    parser.add_argument(
        "--task",
        choices=["auto", "regression", "classification"],
        default="auto",
        help="Target type. auto uses the dtype and the number of distinct values.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eda_report"),
        help="Directory for report.md and figures/ (default: eda_report).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data is None and args.target is None:
        data: Path = CLASSIFICATION_CSV
        target = CLASSIFICATION_TARGET
    elif args.data is None or args.target is None:
        print(
            "error: pass both --data and --target, or neither to use the built-in classification sample.",
            file=sys.stderr,
        )
        return 2
    else:
        data = args.data
        target = args.target
    try:
        report = run_analysis(data, target, task=args.task, out=str(args.out))
    except EDAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
