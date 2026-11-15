"""Fit one classifier, or compare every application and select one."""

from __future__ import annotations

import sys

import pandas as pd

from models import ModelError
from models.cli import build_parser, resolve_inputs
from models.data import load_frame, prepare_split
from models.importance import native_weights, permutation_scores
from models.metrics import (
    baseline_metrics,
    cross_validation_metrics,
    holdout_metrics,
    positive_label,
)
from models.preprocess import make_pipeline
from models.report import write_comparison_report, write_model_report


def _classes_from_series(*series) -> list:
    values = pd.concat([pd.Series(part) for part in series], ignore_index=True)
    return sorted(pd.unique(values), key=str)


def _data_summary(bundle: dict, *, source: str, target: str, test_size: float, seed: int, folds: int) -> dict:
    labels = _classes_from_series(bundle["y_train"], bundle["y_test"])
    return {
        "source": source,
        "target": target,
        "source_rows": bundle["source_rows"],
        "missing_target_rows": bundle["missing_target_rows"],
        "duplicate_rows_removed": bundle["duplicate_rows_removed"],
        "rows_used": bundle["rows_used"],
        "empty_columns": bundle["empty_columns"],
        "numeric": bundle["numeric"],
        "categorical": bundle["categorical"],
        "class_counts": bundle["class_counts"],
        "train_rows": int(len(bundle["y_train"])),
        "test_rows": int(len(bundle["y_test"])),
        "test_size": test_size,
        "seed": seed,
        "cv_folds": folds,
        "positive_label": None
        if positive_label(labels) is None
        else str(positive_label(labels)),
    }


def score_model(name: str, note: str, estimator, bundle: dict, *, folds: int, seed: int) -> dict:
    pipeline = make_pipeline(estimator, bundle["numeric"], bundle["categorical"])
    cv = cross_validation_metrics(
        pipeline,
        bundle["x_train"],
        bundle["y_train"],
        folds=folds,
        seed=seed,
    )
    pipeline.fit(bundle["x_train"], bundle["y_train"])
    predicted = pipeline.predict(bundle["x_test"])
    proba = pipeline.predict_proba(bundle["x_test"]) if hasattr(pipeline, "predict_proba") else None
    classes = list(pipeline.named_steps["model"].classes_)
    holdout = holdout_metrics(bundle["y_test"], predicted, proba, classes)
    return {
        "model": name,
        "note": note,
        "cv": cv,
        "holdout": holdout,
        "baseline": baseline_metrics(bundle["y_train"], bundle["y_test"]),
        "importance": permutation_scores(
            pipeline,
            bundle["x_test"],
            bundle["y_test"],
            seed=seed,
        ),
        "native": native_weights(pipeline, bundle["numeric"], bundle["categorical"]),
    }


def _command(program: str, data, target: str, out, test_size: float, seed: int, folds: int) -> str:
    return (
        f"{program} --data {data} --target {target} --out {out} "
        f"--test-size {test_size} --seed {seed} --cv {folds}"
    )


def run_application(name: str, build, note: str, argv: list[str] | None = None) -> int:
    parser = build_parser(f"Classify a target with {name.replace('_', ' ')}.")
    args = parser.parse_args(argv)
    try:
        data_path, target = resolve_inputs(args)
        frame = load_frame(data_path)
        bundle = prepare_split(frame, target, test_size=args.test_size, seed=args.seed)
        result = score_model(
            name,
            note,
            build(args.seed),
            bundle,
            folds=args.cv,
            seed=args.seed,
        )
        result["data"] = _data_summary(
            bundle,
            source=str(data_path),
            target=target,
            test_size=args.test_size,
            seed=args.seed,
            folds=args.cv,
        )
        result["command"] = _command(
            f"python -m models.{name}",
            data_path,
            target,
            args.out,
            args.test_size,
            args.seed,
            args.cv,
        )
        report = write_model_report(result, args.out)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {report}")
    print(
        f"{name}: holdout accuracy {result['holdout']['accuracy']:.3f}, "
        f"macro F1 {result['holdout']['f1_macro']:.3f}"
    )
    return 0


def select_winner(models: list[dict]) -> tuple[str, str]:
    """Pick the highest mean cross-validated macro F1. Ties use ROC-AUC, then name."""

    def sort_key(row: dict):
        auc = row["cv"]["roc_auc_ovr"]["mean"]
        auc_value = -1.0 if auc is None else float(auc)
        return (-row["cv"]["f1_macro"]["mean"], -auc_value, row["model"])

    ranked = sorted(models, key=sort_key)
    winner = ranked[0]
    if len(ranked) == 1:
        reason = f"`{winner['model']}` is the only fitted model."
        return winner["model"], reason
    second = ranked[1]
    margin = winner["cv"]["f1_macro"]["mean"] - second["cv"]["f1_macro"]["mean"]
    reason = (
        f"`{winner['model']}` has the highest mean cross-validated macro F1 "
        f"({winner['cv']['f1_macro']['mean']:.3f}), "
        f"{margin:.3f} above `{second['model']}` ({second['cv']['f1_macro']['mean']:.3f}). "
        "The holdout was not used to choose it."
    )
    return winner["model"], reason


def run_best(argv: list[str] | None = None) -> int:
    from models.catalog import comparison_models

    parser = build_parser(
        "Fit every classification application, including soft voting, and select the best."
    )
    args = parser.parse_args(argv)
    try:
        data_path, target = resolve_inputs(args)
        frame = load_frame(data_path)
        bundle = prepare_split(frame, target, test_size=args.test_size, seed=args.seed)
        fitted = []
        for spec in comparison_models():
            fitted.append(
                score_model(
                    spec.name,
                    spec.note,
                    spec.build(args.seed),
                    bundle,
                    folds=args.cv,
                    seed=args.seed,
                )
            )
        winner, reason = select_winner(fitted)
        summary = _data_summary(
            bundle,
            source=str(data_path),
            target=target,
            test_size=args.test_size,
            seed=args.seed,
            folds=args.cv,
        )
        report = write_comparison_report(
            {
                "data": summary,
                "models": fitted,
                "winner": winner,
                "selection": reason,
                "command": _command(
                    "python -m models",
                    data_path,
                    target,
                    args.out,
                    args.test_size,
                    args.seed,
                    args.cv,
                ),
            },
            args.out,
        )
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {report}")
    print(f"Selected {winner}")
    return 0
