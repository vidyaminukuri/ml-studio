"""Feature importance on the original columns, plus model-internal weights."""

from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance


def permutation_scores(pipeline, x_test, y_test, *, seed: int, repeats: int = 10) -> list[dict]:
    """Drop in holdout macro F1 when each original column is shuffled."""
    result = permutation_importance(
        pipeline,
        x_test,
        y_test,
        scoring="f1_macro",
        n_repeats=int(repeats),
        random_state=int(seed),
        n_jobs=1,
    )
    rows = []
    for name, mean, std in zip(x_test.columns, result.importances_mean, result.importances_std, strict=True):
        rows.append({"feature": str(name), "mean": float(mean), "std": float(std)})
    rows.sort(key=lambda row: (-row["mean"], row["feature"]))
    return rows


def _source_column(name: str, numeric: list[str], categorical: list[str]) -> str:
    if name.startswith("numeric__"):
        return name.split("numeric__", 1)[1]
    if name.startswith("categorical__"):
        rest = name.split("categorical__", 1)[1]
        matches = [
            column
            for column in categorical
            if rest == column or rest.startswith(column + "_")
        ]
        if matches:
            return max(matches, key=len)
        return rest
    return name


def native_weights(pipeline, numeric: list[str], categorical: list[str]) -> dict | None:
    """Coefficients or impurity importances on the transformed columns."""
    model = pipeline.named_steps["model"]
    prepare = pipeline.named_steps["prepare"]
    names = [str(name) for name in prepare.get_feature_names_out()]
    values = None
    method = None
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        values = np.asarray(importances, dtype=float)
        method = "impurity importance on transformed columns"
    else:
        try:
            coef = np.asarray(model.coef_, dtype=float)
        except (AttributeError, ValueError):
            coef = None
        if coef is not None:
            values = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
            method = "mean absolute coefficient on transformed columns"
    if values is None or values.shape[0] != len(names):
        return None
    rows = []
    for name, value in zip(names, values, strict=True):
        rows.append(
            {
                "feature": name,
                "source": _source_column(name, numeric, categorical),
                "value": float(value),
            }
        )
    rows.sort(key=lambda row: (-abs(row["value"]), row["feature"]))
    return {"method": method, "rows": rows[:15]}
