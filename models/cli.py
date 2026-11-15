"""Shared command line arguments for the classification applications."""

from __future__ import annotations

import argparse
from pathlib import Path

from models import ModelError

SAMPLE_CSV = (
    Path(__file__).resolve().parents[1] / "eda" / "data" / "sample_classification.csv"
)
SAMPLE_TARGET = "churn"


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="CSV path. Omit, together with --target, to use the classification sample.",
    )
    parser.add_argument("--target", default=None, help="Column to predict.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("model_report"),
        help="Directory for report.md and figures/ (default: model_report).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of rows held out for validation (default: 0.25).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split and model seed.")
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Stratified folds on the training rows (default: 5).",
    )
    return parser


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, str]:
    if args.cv < 2:
        raise ModelError("--cv must be at least 2.")
    if args.data is None and args.target is None:
        return SAMPLE_CSV, SAMPLE_TARGET
    if args.data is None or args.target is None:
        raise ModelError(
            "Pass both --data and --target, or neither to use the classification sample."
        )
    return args.data, args.target
