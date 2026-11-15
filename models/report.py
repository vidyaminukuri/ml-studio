"""Markdown report and plots for a classification run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

def _num(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _cell(value) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    if len(text) > 48:
        return text[:45] + "..."
    return text


def _table(headers: list[str], rows: list[list]) -> list[str]:
    if not rows:
        return ["_None._"]
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return [head, rule, *body]


def _prepare(out_dir: Path) -> Path:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for path in figures.glob("*"):
        if path.is_file():
            path.unlink()
    return figures


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{path.name}"


def _plot_confusion(metrics: dict, figures: Path) -> str:
    labels = metrics["confusion"]["labels"]
    matrix = np.asarray(metrics["confusion"]["matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 3.2, 1.2 * len(labels) + 2.4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Holdout confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046)
    return _save(fig, figures / "confusion_matrix.png")


def _plot_importance(rows: list[dict], figures: Path, stem: str = "feature_importance") -> str | None:
    if not rows:
        return None
    shown = list(reversed(rows[:12]))
    fig, ax = plt.subplots(figsize=(7.4, max(3.0, 0.38 * len(shown) + 1.2)))
    ax.barh(
        [row["feature"] for row in shown],
        [row["mean"] for row in shown],
        xerr=[row["std"] for row in shown],
        color="#3D5A80",
    )
    ax.set_xlabel("Holdout macro F1 drop when shuffled")
    ax.set_title("Permutation importance")
    return _save(fig, figures / f"{stem}.png")


def _plot_leaderboard(models: list[dict], figures: Path) -> str:
    ranked = sorted(models, key=lambda row: row["cv"]["f1_macro"]["mean"])
    fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.42 * len(ranked) + 1.2)))
    means = [row["cv"]["f1_macro"]["mean"] for row in ranked]
    stds = [row["cv"]["f1_macro"]["std"] for row in ranked]
    ax.barh([row["model"] for row in ranked], means, xerr=stds, color="#3D5A80")
    ax.set_xlabel("Cross-validated macro F1")
    ax.set_xlim(0, 1)
    ax.set_title("Model comparison")
    return _save(fig, figures / "leaderboard.png")


def _metric_rows(metrics: dict) -> list[list]:
    return [
        ["accuracy", _num(metrics["accuracy"])],
        ["precision, macro", _num(metrics["precision_macro"])],
        ["recall, macro", _num(metrics["recall_macro"])],
        ["F1, macro", _num(metrics["f1_macro"])],
        ["precision, weighted", _num(metrics["precision_weighted"])],
        ["recall, weighted", _num(metrics["recall_weighted"])],
        ["F1, weighted", _num(metrics["f1_weighted"])],
        ["ROC-AUC", _num(metrics["roc_auc"])],
        ["log loss", _num(metrics["log_loss"])],
    ]


def _cv_rows(cv: dict) -> list[list]:
    labels = [
        ("accuracy", "accuracy"),
        ("precision_macro", "precision, macro"),
        ("recall_macro", "recall, macro"),
        ("f1_macro", "F1, macro"),
        ("f1_weighted", "F1, weighted"),
        ("roc_auc_ovr", "ROC-AUC"),
    ]
    rows = []
    for key, label in labels:
        stats = cv.get(key) or {}
        rows.append([label, _num(stats.get("mean")), _num(stats.get("std"))])
    return rows


def _data_lines(data: dict) -> list[str]:
    lines = [
        "## Data",
        "",
        f"Source: `{data['source']}`.",
        f"Target: `{data['target']}`.",
        f"Rows loaded: {data['source_rows']}. "
        f"Rows with a missing target removed: {data['missing_target_rows']}. "
        f"Exact duplicate rows removed: {data['duplicate_rows_removed']}. "
        f"Rows used: {data['rows_used']}.",
        f"Training rows: {data['train_rows']}. Holdout rows: {data['test_rows']}. "
        f"Test fraction: {data['test_size']}. Seed: {data['seed']}. "
        f"Stratified folds: {data['cv_folds']}.",
        "",
        "Numeric features: " + (", ".join(f"`{name}`" for name in data["numeric"]) or "none") + ".",
        "Categorical features: "
        + (", ".join(f"`{name}`" for name in data["categorical"]) or "none")
        + ".",
        "",
        "Class counts on the rows used for the split:",
        "",
    ]
    total = sum(count for _, count in data["class_counts"])
    lines.extend(
        _table(
            ["class", "rows", "share"],
            [
                [label, count, _num(count / total if total else None)]
                for label, count in data["class_counts"]
            ],
        )
    )
    if data["empty_columns"]:
        names = ", ".join(f"`{name}`" for name in data["empty_columns"])
        lines.extend(["", f"Columns dropped because every value was missing: {names}."])
    positive = data.get("positive_label")
    lines.extend(
        [
            "",
            "Empty feature cells are imputed inside the training pipeline: median for numeric "
            "columns and the most frequent value for categorical columns. Numeric columns are "
            "standardized. Categorical columns are one-hot encoded. The holdout rows are not "
            "used to fit those steps.",
            "",
        ]
    )
    if positive:
        lines.append(
            f"ROC-AUC is the ranking score for class `{positive}` "
            "(the later of the two labels in sorted order)."
        )
    else:
        lines.append("ROC-AUC is the macro average of one-vs-rest scores.")
    return lines


def _model_section(result: dict, figures: Path) -> list[str]:
    holdout = result["holdout"]
    lines = [
        f"## {result['model']}",
        "",
        result["note"],
        "",
        "### Cross-validation on the training rows",
        "",
    ]
    lines.extend(_table(["metric", "mean", "std across folds"], _cv_rows(result["cv"])))
    lines.extend(["", "### Holdout", ""])
    lines.extend(_table(["metric", "value"], _metric_rows(holdout)))
    lines.extend(["", "### Per class", ""])
    lines.extend(
        _table(
            ["class", "precision", "recall", "F1", "support"],
            [
                [
                    row["label"],
                    _num(row["precision"]),
                    _num(row["recall"]),
                    _num(row["f1"]),
                    row["support"],
                ]
                for row in holdout["per_class"]
            ],
        )
    )
    lines.extend(["", "### Confusion matrix", ""])
    labels = holdout["confusion"]["labels"]
    matrix_rows = []
    for label, row in zip(labels, holdout["confusion"]["matrix"], strict=True):
        matrix_rows.append([label, *row])
    lines.extend(_table(["actual \\ predicted", *labels], matrix_rows))
    confusion_name = _plot_confusion(holdout, figures)
    lines.extend(["", f"![Holdout confusion matrix]({confusion_name})", ""])
    baseline = result["baseline"]
    lines.extend(
        [
            "### Majority-class baseline on the same holdout",
            "",
            f"Every holdout row is predicted as `{baseline['predicted_class']}`.",
            "",
        ]
    )
    lines.extend(
        _table(
            ["metric", "baseline", "model"],
            [
                ["accuracy", _num(baseline["accuracy"]), _num(holdout["accuracy"])],
                ["precision, macro", _num(baseline["precision_macro"]), _num(holdout["precision_macro"])],
                ["recall, macro", _num(baseline["recall_macro"]), _num(holdout["recall_macro"])],
                ["F1, macro", _num(baseline["f1_macro"]), _num(holdout["f1_macro"])],
            ],
        )
    )
    lines.extend(["", "### Permutation importance", ""])
    lines.append(
        "Each value is the drop in holdout macro F1 when that original column is shuffled. "
        "Ten shuffles are averaged. A negative value means the shuffled column scored higher."
    )
    lines.append("")
    lines.extend(
        _table(
            ["feature", "mean F1 drop", "std"],
            [[row["feature"], _num(row["mean"], 4), _num(row["std"], 4)] for row in result["importance"]],
        )
    )
    chart = _plot_importance(result["importance"], figures, stem="feature_importance")
    if chart:
        lines.extend(["", f"![Permutation importance]({chart})", ""])
    native = result.get("native")
    if native:
        lines.extend(
            [
                "### Model-internal weights",
                "",
                native["method"] + ". These are on the transformed columns, so a standardized "
                "numeric feature and a one-hot level are not in the same unit.",
                "",
            ]
        )
        lines.extend(
            _table(
                ["transformed feature", "source column", "weight"],
                [
                    [row["feature"], row["source"], _num(row["value"], 4)]
                    for row in native["rows"]
                ],
            )
        )
    return lines


def write_model_report(result: dict, out_dir: str | Path) -> Path:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figures = _prepare(destination)
    lines = [
        f"# {result['model']}",
        "",
        result["command"],
        "",
        *_data_lines(result["data"]),
        "",
        *_model_section(result, figures),
        "",
    ]
    path = destination / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _holdout_check(winner: dict) -> list[str]:
    """Say how the selected model compares with the majority-class rule on the holdout."""
    holdout = winner["holdout"]
    baseline = winner["baseline"]
    lines = []
    if holdout["accuracy"] < baseline["accuracy"]:
        lines.append(
            f"On this holdout, accuracy {_num(holdout['accuracy'])} is below the "
            f"majority-class accuracy {_num(baseline['accuracy'])}."
        )
    else:
        lines.append(
            f"On this holdout, accuracy {_num(holdout['accuracy'])} is at least the "
            f"majority-class accuracy {_num(baseline['accuracy'])}."
        )
    if holdout["f1_macro"] > baseline["f1_macro"]:
        lines.append(
            f"Holdout macro F1 {_num(holdout['f1_macro'])} is above the majority-class "
            f"macro F1 {_num(baseline['f1_macro'])}."
        )
    else:
        lines.append(
            f"Holdout macro F1 {_num(holdout['f1_macro'])} is not above the majority-class "
            f"macro F1 {_num(baseline['f1_macro'])}."
        )
    auc = holdout["roc_auc"]
    cv_auc = winner["cv"]["roc_auc_ovr"]["mean"]
    if auc is not None and auc < 0.5:
        lines.append(f"Holdout ROC-AUC is {_num(auc)}, below 0.5 on this split.")
    if auc is not None and cv_auc is not None and cv_auc - auc > 0.05:
        lines.append(
            f"Cross-validated ROC-AUC ({_num(cv_auc)}) is higher than the holdout "
            f"ROC-AUC ({_num(auc)})."
        )
    return lines


def write_comparison_report(result: dict, out_dir: str | Path) -> Path:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figures = _prepare(destination)
    models = sorted(
        result["models"],
        key=lambda row: (-row["cv"]["f1_macro"]["mean"], row["model"]),
    )
    winner = next(row for row in models if row["model"] == result["winner"])
    lines = [
        "# Classification model comparison",
        "",
        result["command"],
        "",
        "Hyperparameters are fixed. They are not searched.",
        "",
        *_data_lines(result["data"]),
        "",
        "## Leaderboard",
        "",
        f"Selected model: `{result['winner']}`.",
        result["selection"],
        "",
        *_holdout_check(winner),
        "",
    ]
    board = []
    for row in models:
        board.append(
            [
                row["model"],
                _num(row["cv"]["f1_macro"]["mean"]),
                _num(row["cv"]["f1_macro"]["std"]),
                _num(row["cv"]["accuracy"]["mean"]),
                _num(row["cv"]["roc_auc_ovr"]["mean"]),
                _num(row["holdout"]["accuracy"]),
                _num(row["holdout"]["precision_macro"]),
                _num(row["holdout"]["recall_macro"]),
                _num(row["holdout"]["f1_macro"]),
                _num(row["holdout"]["roc_auc"]),
            ]
        )
    lines.extend(
        _table(
            [
                "model",
                "CV F1",
                "CV F1 std",
                "CV accuracy",
                "CV ROC-AUC",
                "holdout accuracy",
                "holdout precision",
                "holdout recall",
                "holdout F1",
                "holdout ROC-AUC",
            ],
            board,
        )
    )
    lines.append("")
    lines.append("Precision, recall, and F1 on this table are macro averages.")
    chart = _plot_leaderboard(models, figures)
    lines.extend(["", f"![Cross-validated macro F1]({chart})", ""])
    lines.extend(
        [
            "## Selected model",
            "",
            "The sections below are the selected model's validation detail.",
            "",
        ]
    )
    # Confusion and importance charts are written from the winner only.
    detail = _model_section(winner, figures)
    lines.extend(detail)
    lines.append("")
    path = destination / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
