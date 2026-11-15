"""Holdout and cross-validation metrics for a fitted classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

CV_SCORING = {
    "accuracy": "accuracy",
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    "roc_auc_ovr": "roc_auc_ovr",
}


def _sorted_labels(values) -> list:
    return sorted(pd.unique(values), key=str)


def positive_label(labels: list) -> object | None:
    """For two classes, ROC-AUC uses the later label in sorted order."""
    if len(labels) == 2:
        return labels[1]
    return None


def roc_auc(y_true, proba, classes: list) -> float | None:
    if proba is None or len(classes) < 2:
        return None
    y_true = np.asarray(y_true)
    try:
        if len(classes) == 2:
            index = list(classes).index(classes[1])
            return float(roc_auc_score(y_true == classes[1], proba[:, index]))
        return float(
            roc_auc_score(
                y_true,
                proba,
                multi_class="ovr",
                average="macro",
                labels=list(classes),
            )
        )
    except ValueError:
        return None


def holdout_metrics(y_true, y_pred, proba, classes: list) -> dict:
    labels = list(classes)
    per_class = []
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    actual = list(np.asarray(y_true))
    for label, prec, rec, f_one in zip(labels, precision, recall, f1, strict=True):
        per_class.append(
            {
                "label": str(label),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f_one),
                "support": int(sum(value == label for value in actual)),
            }
        )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    loss = None
    if proba is not None:
        try:
            loss = float(log_loss(y_true, proba, labels=labels))
        except ValueError:
            loss = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": roc_auc(y_true, proba, labels),
        "log_loss": loss,
        "per_class": per_class,
        "confusion": {
            "labels": [str(label) for label in labels],
            "matrix": matrix.astype(int).tolist(),
        },
    }


def baseline_metrics(y_train, y_test) -> dict:
    """Predict the training majority class for every holdout row."""
    mode = pd.Series(y_train).mode().iloc[0]
    predicted = np.full(len(y_test), mode)
    classes = _sorted_labels(pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True))
    metrics = holdout_metrics(y_test, predicted, None, classes)
    metrics["predicted_class"] = str(mode)
    metrics["roc_auc"] = 0.5 if len(classes) == 2 else None
    return metrics


def cross_validation_metrics(pipeline, x_train, y_train, *, folds: int, seed: int) -> dict:
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    scoring = dict(CV_SCORING)
    try:
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
    except ValueError:
        scoring.pop("roc_auc_ovr", None)
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
    summary = {}
    for key in scoring:
        values = np.asarray(scores[f"test_{key}"], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    if "roc_auc_ovr" not in summary:
        summary["roc_auc_ovr"] = {"mean": None, "std": None}
    return summary
