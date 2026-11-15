"""Statistics for a generic table with one target column.

The analysis path does not know any dataset's column names. Callers pass a
frame and the target column. Task detection, when requested, uses only the
target's dtype and how many distinct non-null values it has.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# A numeric target with this many distinct non-null values, or fewer, is
# classification. One more distinct value makes it regression.
AUTO_MAX_DISTINCT = 10
NEAR_CONSTANT_SHARE = 0.95
IMBALANCE_SHARE = 0.70
SKEW_FLAG = 1.0
IQR_K = 1.5
RARE_LEVEL_SHARE = 0.01
HIGH_CARDINALITY = 20
HIGH_CARDINALITY_MIN_ROWS = 10
CORR_FLAG = 0.7
TOP_LEVELS = 10
RARE_LIST_LIMIT = 15
PREVIEW_ROWS = 5
PREVIEW_COLS = 12
SCATTER_MAX = 5000
MISSING_CHART_MAX = 40
PLOT_PAGE = 9


class EDAError(ValueError):
    """The table cannot be analyzed as requested."""


def format_number(value: object, digits: int = 3) -> str:
    if value is None or isinstance(value, bool):
        return "n/a"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def format_percent(rate: object) -> str:
    if rate is None or isinstance(rate, bool):
        return "n/a"
    try:
        number = float(rate)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number * 100:.1f}%"


def label(value: object) -> str:
    """Stable display label. Does not merge unequal numeric values."""
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return str(value)
        if number.is_integer():
            return str(int(number))
        return format(number, ".12g")
    return str(value)


def is_numeric(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    )


@dataclass(frozen=True)
class TaskDecision:
    task: str
    reason: str
    source: str


def detect_task(series: pd.Series, max_distinct: int = AUTO_MAX_DISTINCT) -> TaskDecision:
    """Choose classification or regression from the target column alone.

    * A boolean, categorical, or string-like target is classification.
    * A numeric target is classification when it has at most ``max_distinct``
      distinct non-null values, and regression otherwise. Missing values are
      not a level.
    * A target with no non-null values follows its storage type: numeric
      becomes regression, and anything else becomes classification.
    """
    observed = series.dropna()
    n_distinct = int(observed.nunique(dropna=True))
    if len(observed) == 0:
        if is_numeric(series):
            return TaskDecision(
                "regression",
                "The target has no non-null values and is numeric, so the task is regression.",
                "auto",
            )
        return TaskDecision(
            "classification",
            "The target has no non-null values and is not numeric, so the task is classification.",
            "auto",
        )
    if not is_numeric(series):
        return TaskDecision(
            "classification",
            (
                f"The target dtype is {series.dtype}, which is not numeric, "
                "so the task is classification."
            ),
            "auto",
        )
    if n_distinct <= max_distinct:
        return TaskDecision(
            "classification",
            (
                f"The target is numeric with {n_distinct} distinct non-null values, "
                f"which is at most {max_distinct}, so the task is classification."
            ),
            "auto",
        )
    return TaskDecision(
        "regression",
        (
            f"The target is numeric with {n_distinct} distinct non-null values, "
            f"which is more than {max_distinct}, so the task is regression."
        ),
        "auto",
    )


def resolve_task(series: pd.Series, task: str) -> TaskDecision:
    if task == "auto":
        return detect_task(series)
    if task not in {"classification", "regression"}:
        raise EDAError("task must be auto, classification, or regression.")
    return TaskDecision(
        task,
        f"The task was set to {task} explicitly.",
        "explicit",
    )


def load_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV. Fail clearly if the file is missing, empty, or header-only."""
    source = Path(path)
    if not source.exists():
        raise EDAError(f"Data file not found: {source}")
    if not source.is_file():
        raise EDAError(f"Data path is not a file: {source}")
    if source.stat().st_size == 0:
        raise EDAError(f"Data file is empty: {source}")
    try:
        frame = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise EDAError(f"Data file is empty: {source}") from exc
    except pd.errors.ParserError as exc:
        raise EDAError(f"Could not parse CSV {source}: {exc}") from exc
    if frame.shape[1] == 0 or frame.shape[0] == 0:
        raise EDAError(f"Data file has no rows: {source}")
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        names = ", ".join(repr(str(name)) for name in duplicated)
        raise EDAError(f"Data file has duplicate column names: {names}")
    return frame


def require_target(frame: pd.DataFrame, target: str) -> None:
    if target is None or str(target).strip() == "":
        raise EDAError("A target column is required.")
    if target not in frame.columns:
        available = ", ".join(str(column) for column in frame.columns)
        raise EDAError(
            f"Target column {target!r} is not in the data. Available columns: {available}"
        )


def missingness(frame: pd.DataFrame) -> dict:
    """Column and row missing counts. Rates use the full row count as the denominator."""
    n_rows = int(frame.shape[0])
    if n_rows == 0:
        raise EDAError("Data file has no rows.")
    mask = frame.isna()
    columns = []
    for name in frame.columns:
        count = int(mask[name].sum())
        columns.append(
            {"column": str(name), "count": count, "rate": count / n_rows}
        )
    per_row = mask.sum(axis=1).astype(int)
    max_missing = int(per_row.max()) if n_rows else 0
    histogram = []
    for n_missing in range(max_missing + 1):
        rows = int((per_row == n_missing).sum())
        histogram.append(
            {"n_missing": n_missing, "rows": rows, "rate": rows / n_rows}
        )
    rows_with_any = int((per_row > 0).sum())
    n_cells = n_rows * int(frame.shape[1])
    n_missing_cells = int(mask.to_numpy().sum())
    return {
        "n_rows": n_rows,
        "n_cols": int(frame.shape[1]),
        "n_cells": n_cells,
        "n_missing_cells": n_missing_cells,
        "cell_rate": (n_missing_cells / n_cells) if n_cells else None,
        "columns": columns,
        "rows_with_any": rows_with_any,
        "rows_with_any_rate": rows_with_any / n_rows,
        "max_missing_in_row": max_missing,
        "row_histogram": histogram,
    }


def _finite(series: pd.Series) -> np.ndarray:
    number = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return number[np.isfinite(number)]


def skewness(values: np.ndarray) -> float | None:
    """Bias-corrected Fisher-Pearson skew, matching ``Series.skew``."""
    n = int(values.size)
    if n < 3:
        return None
    centered = values - values.mean()
    m2 = float(np.mean(centered**2))
    if m2 == 0.0:
        return None
    m3 = float(np.mean(centered**3))
    g1 = m3 / (m2**1.5)
    return float((np.sqrt(n * (n - 1)) / (n - 2)) * g1)


def numeric_profile(series: pd.Series) -> dict:
    """Univariate stats for one numeric column, plus Tukey IQR fences."""
    number = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    missing = int(pd.isna(series).sum())
    non_finite = int(np.sum(~np.isfinite(number) & ~np.isnan(number)))
    values = number[np.isfinite(number)]
    count = int(values.size)
    profile = {
        "count": count,
        "missing": missing,
        "non_finite": non_finite,
        "mean": None,
        "std": None,
        "min": None,
        "q1": None,
        "median": None,
        "q3": None,
        "max": None,
        "skew": None,
        "lower_fence": None,
        "upper_fence": None,
        "outlier_count": 0,
        "outlier_rate": None,
    }
    if count == 0:
        return profile
    q1, median, q3 = (float(x) for x in np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    lower = q1 - IQR_K * iqr
    upper = q3 + IQR_K * iqr
    outlier_count = int(np.sum((values < lower) | (values > upper)))
    profile.update(
        {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if count >= 2 else None,
            "min": float(values.min()),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": float(values.max()),
            "skew": skewness(values),
            "lower_fence": float(lower),
            "upper_fence": float(upper),
            "outlier_count": outlier_count,
            "outlier_rate": outlier_count / count,
        }
    )
    return profile


def _is_high_cardinality(cardinality: int, n_non_null: int) -> bool:
    if cardinality >= HIGH_CARDINALITY:
        return True
    return n_non_null >= HIGH_CARDINALITY_MIN_ROWS and cardinality == n_non_null


def categorical_profile(series: pd.Series) -> dict:
    missing = int(series.isna().sum())
    n_rows = int(len(series))
    observed = series.dropna()
    n_non_null = int(len(observed))
    top: list[dict] = []
    rare: list[dict] = []
    if n_non_null == 0:
        return {
            "missing": missing,
            "missing_rate": missing / n_rows if n_rows else None,
            "n_non_null": 0,
            "cardinality": 0,
            "high_cardinality": False,
            "top": top,
            "n_rare": 0,
            "rare": rare,
        }
    counts = observed.value_counts(dropna=True)
    levels = []
    for value, count in counts.items():
        share = float(count) / n_non_null
        levels.append(
            {"level": label(value), "count": int(count), "rate": share}
        )
    levels.sort(key=lambda row: (-row["count"], row["level"]))
    rare_all = [row for row in levels if row["rate"] <= RARE_LEVEL_SHARE]
    rare_all.sort(key=lambda row: (row["count"], row["level"]))
    return {
        "missing": missing,
        "missing_rate": missing / n_rows if n_rows else None,
        "n_non_null": n_non_null,
        "cardinality": len(levels),
        "high_cardinality": _is_high_cardinality(len(levels), n_non_null),
        "top": levels[:TOP_LEVELS],
        "n_rare": len(rare_all),
        "rare": rare_all[:RARE_LIST_LIMIT],
    }


def _class_rows(series: pd.Series) -> list[dict]:
    observed = series.dropna()
    n_non_null = int(len(observed))
    if n_non_null == 0:
        return []
    counts = observed.value_counts(dropna=True)
    rows = []
    for value, count in counts.items():
        rows.append(
            {
                "level": label(value),
                "count": int(count),
                "rate": float(count) / n_non_null,
            }
        )
    rows.sort(key=lambda row: (-row["count"], row["level"]))
    return rows


def target_profile(series: pd.Series, task: str) -> dict:
    n_rows = int(len(series))
    missing = int(series.isna().sum())
    classes = _class_rows(series)
    n_non_null = int(series.notna().sum())
    n_distinct = len(classes)
    if classes:
        majority = classes[0]
        majority_level = majority["level"]
        majority_count = majority["count"]
        majority_share = majority["rate"]
    else:
        majority_level = None
        majority_count = None
        majority_share = None
    constant = n_distinct <= 1
    near_constant = (
        not constant
        and majority_share is not None
        and majority_share >= NEAR_CONSTANT_SHARE
    )
    imbalanced = (
        task == "classification"
        and not constant
        and not near_constant
        and majority_share is not None
        and majority_share >= IMBALANCE_SHARE
    )
    profile = {
        "missing": missing,
        "missing_rate": missing / n_rows if n_rows else None,
        "n_non_null": n_non_null,
        "n_distinct": n_distinct,
        "constant": constant,
        "near_constant": near_constant,
        "imbalanced": imbalanced,
        "majority_level": majority_level,
        "majority_count": majority_count,
        "majority_share": majority_share,
        "classes": classes,
        "numeric": numeric_profile(series) if task == "regression" else None,
    }
    return profile


def _feature_level(value: object) -> str:
    if value is None or value is pd.NA:
        return "<missing>"
    try:
        if bool(pd.isna(value)):
            return "<missing>"
    except (TypeError, ValueError):
        pass
    return label(value)


def _associations(frame: pd.DataFrame, target: str, task: str) -> dict:
    target_series = frame[target]
    usable = frame.loc[target_series.notna()].copy()
    numeric_rows = []
    categorical_rows = []
    class_order = [row["level"] for row in _class_rows(target_series)]
    for name in frame.columns:
        if name == target:
            continue
        series = usable[name]
        if is_numeric(frame[name]):
            numeric_rows.append(
                _numeric_association(series, usable[target], str(name), task, class_order)
            )
        else:
            categorical_rows.append(
                _categorical_association(series, usable[target], str(name), task, class_order)
            )
    return {"numeric": numeric_rows, "categorical": categorical_rows}


def _numeric_association(
    feature: pd.Series,
    target: pd.Series,
    name: str,
    task: str,
    class_order: list[str],
) -> dict:
    values = pd.to_numeric(feature, errors="coerce").to_numpy(dtype="float64")
    if task == "regression":
        target_values = pd.to_numeric(target, errors="coerce").to_numpy(dtype="float64")
        mask = np.isfinite(values) & np.isfinite(target_values)
        x = values[mask]
        y = target_values[mask]
        pearson = None
        if x.size >= 2 and float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
            pearson = float(np.corrcoef(x, y)[0, 1])
        return {"column": name, "n": int(x.size), "pearson_r": pearson}
    labels = target.map(label)
    by_class = []
    for level in class_order:
        group = values[labels.to_numpy() == level]
        finite = group[np.isfinite(group)]
        by_class.append(
            {
                "level": level,
                "n": int(finite.size),
                "mean": float(finite.mean()) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
            }
        )
    return {"column": name, "by_class": by_class}


def _categorical_association(
    feature: pd.Series,
    target: pd.Series,
    name: str,
    task: str,
    class_order: list[str],
) -> dict:
    levels = feature.map(_feature_level)
    if task == "classification":
        grouped: dict[str, dict[str, int]] = {}
        target_labels = target.map(label)
        for level, target_level in zip(levels, target_labels, strict=True):
            bucket = grouped.setdefault(str(level), {})
            bucket[str(target_level)] = bucket.get(str(target_level), 0) + 1
        rows = []
        for level, counts in grouped.items():
            n = int(sum(counts.values()))
            classes = []
            for class_name in class_order:
                count = int(counts.get(class_name, 0))
                classes.append(
                    {
                        "level": class_name,
                        "count": count,
                        "rate": (count / n) if n else None,
                    }
                )
            rows.append({"level": level, "n": n, "classes": classes})
    else:
        target_values = pd.to_numeric(target, errors="coerce")
        buckets: dict[str, list[float]] = {}
        for level, target_value in zip(levels, target_values, strict=True):
            if not np.isfinite(target_value):
                continue
            buckets.setdefault(str(level), []).append(float(target_value))
        rows = []
        for level, values_list in buckets.items():
            values = np.asarray(values_list, dtype="float64")
            rows.append(
                {
                    "level": level,
                    "n": int(values.size),
                    "mean": float(values.mean()) if values.size else None,
                    "median": float(np.median(values)) if values.size else None,
                }
            )
    rows.sort(key=lambda row: (-row["n"], row["level"]))
    missing_row = next((row for row in rows if row["level"] == "<missing>"), None)
    shown = rows[:TOP_LEVELS]
    shown_names = {row["level"] for row in shown}
    if missing_row is not None and missing_row["level"] not in shown_names:
        shown.append(missing_row)
        shown_names.add(missing_row["level"])
    omitted = len(rows) - len(shown)
    return {"column": name, "omitted_levels": omitted, "rows": shown}


def _correlation(frame: pd.DataFrame) -> dict:
    columns = [str(name) for name in frame.columns if is_numeric(frame[name])]
    if len(columns) < 2:
        return {"columns": columns, "matrix": [], "high_pairs": []}
    numbers = frame[columns].apply(pd.to_numeric, errors="coerce")
    corr = numbers.corr(method="pearson", min_periods=2)
    matrix = []
    for left in columns:
        row = []
        for right in columns:
            value = corr.loc[left, right]
            row.append(None if pd.isna(value) else float(value))
        matrix.append(row)
    high_pairs = []
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            value = matrix[i][j]
            if value is not None and abs(value) >= CORR_FLAG:
                high_pairs.append({"a": left, "b": columns[j], "r": value})
    high_pairs.sort(key=lambda pair: (-abs(pair["r"]), pair["a"], pair["b"]))
    return {"columns": columns, "matrix": matrix, "high_pairs": high_pairs}


def _duplicates(frame: pd.DataFrame) -> dict:
    n_rows = int(frame.shape[0])
    extra = int(frame.duplicated(keep="first").sum())
    in_groups = int(frame.duplicated(keep=False).sum())
    return {
        "extra": extra,
        "in_groups": in_groups,
        "extra_rate": extra / n_rows if n_rows else None,
    }


def _plain(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    return str(value)


def _overview(frame: pd.DataFrame, target: str) -> dict:
    columns = []
    for name in frame.columns:
        series = frame[name]
        columns.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "distinct": int(series.nunique(dropna=True)),
                "kind": "numeric" if is_numeric(series) else "categorical",
                "role": "target" if name == target else "feature",
            }
        )
    preview_columns = [str(name) for name in list(frame.columns[:PREVIEW_COLS])]
    preview_rows = []
    for record in frame.loc[:, preview_columns].head(PREVIEW_ROWS).to_dict(orient="records"):
        preview_rows.append({column: _plain(record[column]) for column in preview_columns})
    omitted = [str(name) for name in list(frame.columns[PREVIEW_COLS:])]
    return {
        "columns": columns,
        "preview_columns": preview_columns,
        "preview_rows": preview_rows,
        "preview_omitted_columns": omitted,
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
    }


def _join_named(parts: list[str]) -> str:
    return "; ".join(parts)


def _findings(result: dict) -> list[str]:
    n_rows = result["n_rows"]
    lines = []
    duplicates = result["duplicates"]
    if duplicates["extra"] == 0:
        lines.append("No exact duplicate rows.")
    else:
        lines.append(
            f"{duplicates['extra']} rows are exact copies of an earlier row "
            f"({format_percent(duplicates['extra_rate'])} of {n_rows} rows). "
            f"{duplicates['in_groups']} rows belong to a duplicated group."
        )

    missing_columns = [row for row in result["missing"]["columns"] if row["count"] > 0]
    missing_columns.sort(key=lambda row: (-row["rate"], row["column"]))
    if not missing_columns:
        lines.append("No column has missing values.")
    else:
        parts = [
            f"{row['column']} ({row['count']}/{n_rows}, {format_percent(row['rate'])})"
            for row in missing_columns
        ]
        lines.append("Missing values: " + _join_named(parts) + ".")
    if result["missing"]["rows_with_any"] == 0:
        lines.append("No row has a missing value.")
    else:
        lines.append(
            f"{result['missing']['rows_with_any']} of {n_rows} rows have at least one missing value "
            f"({format_percent(result['missing']['rows_with_any_rate'])}). "
            f"The most missing cells in a single row is {result['missing']['max_missing_in_row']}."
        )

    lines.extend(_target_findings(result))

    skewed = [
        column
        for column in result["numeric"]
        if column["skew"] is not None and abs(column["skew"]) >= SKEW_FLAG
    ]
    skewed.sort(key=lambda column: (-abs(column["skew"]), column["column"]))
    if not skewed:
        lines.append(f"No numeric feature has |skew| >= {SKEW_FLAG:g}.")
    else:
        parts = []
        for column in skewed:
            direction = "right-skewed" if column["skew"] > 0 else "left-skewed"
            parts.append(
                f"{column['column']} is {direction} (skew {format_number(column['skew'])})"
            )
        lines.append("Skewed numeric features: " + _join_named(parts) + ".")

    outliers = [column for column in result["numeric"] if column["outlier_count"] > 0]
    outliers.sort(key=lambda column: (-column["outlier_count"], column["column"]))
    if not outliers:
        lines.append("No numeric feature has IQR outliers.")
    else:
        parts = [
            (
                f"{column['column']} has {column['outlier_count']} of {column['count']} "
                f"finite values outside the fences ({format_percent(column['outlier_rate'])})"
            )
            for column in outliers
        ]
        lines.append("IQR outliers: " + _join_named(parts) + ".")

    high_card = [column for column in result["categorical"] if column["high_cardinality"]]
    high_card.sort(key=lambda column: (-column["cardinality"], column["column"]))
    if not high_card:
        lines.append(
            "No categorical feature is high-cardinality "
            f"({HIGH_CARDINALITY} or more levels, or an identifier with at least "
            f"{HIGH_CARDINALITY_MIN_ROWS} non-null rows)."
        )
    else:
        parts = [
            f"{column['column']} ({column['cardinality']} levels)" for column in high_card
        ]
        lines.append("High-cardinality categorical features: " + _join_named(parts) + ".")

    rare_columns = [column for column in result["categorical"] if column["n_rare"] > 0]
    rare_columns.sort(key=lambda column: (-column["n_rare"], column["column"]))
    if not rare_columns:
        lines.append(
            f"No categorical feature has a rare level (share <= {RARE_LEVEL_SHARE:.0%})."
        )
    else:
        parts = []
        for column in rare_columns:
            listed = column["rare"][:6]
            detail = ", ".join(
                f"{row['level']} ({row['count']} {'row' if row['count'] == 1 else 'rows'})"
                for row in listed
            )
            extra = column["n_rare"] - len(listed)
            if extra > 0:
                detail = f"{detail}, and {extra} more"
            parts.append(f"{column['column']} ({column['n_rare']}): {detail}")
        lines.append("Rare levels: " + _join_named(parts) + ".")

    correlation = result["correlation"]
    if len(correlation["columns"]) < 2:
        lines.append("Fewer than two numeric columns, so no Pearson correlation was computed.")
    elif not correlation["high_pairs"]:
        lines.append(f"No numeric pair has |Pearson r| >= {CORR_FLAG:g}.")
    else:
        parts = [
            f"{pair['a']} and {pair['b']} (r = {format_number(pair['r'])})"
            for pair in correlation["high_pairs"]
        ]
        lines.append("High correlations: " + _join_named(parts) + ".")
    return lines


def _target_findings(result: dict) -> list[str]:
    profile = result["target_profile"]
    name = result["target"]
    n_rows = result["n_rows"]
    lines = []
    if profile["missing"] > 0:
        lines.append(
            f"Target `{name}` is missing {profile['missing']}/{n_rows} "
            f"({format_percent(profile['missing_rate'])})."
        )
    if profile["n_non_null"] == 0:
        lines.append(f"Target `{name}` has no non-null values.")
        return lines
    if profile["constant"]:
        level = profile["majority_level"]
        count = profile["majority_count"]
        lines.append(
            f"Target `{name}` is constant: every non-null value is {level} "
            f"({count}/{profile['n_non_null']})."
        )
        return lines
    if result["task"] == "classification":
        if profile["near_constant"]:
            lines.append(
                f"Target `{name}` is near-constant: {profile['majority_level']} is "
                f"{profile['majority_count']}/{profile['n_non_null']} non-null rows "
                f"({format_percent(profile['majority_share'])})."
            )
        elif profile["imbalanced"]:
            lines.append(
                f"Target `{name}` is imbalanced: {profile['majority_level']} is "
                f"{profile['majority_count']}/{profile['n_non_null']} non-null rows "
                f"({format_percent(profile['majority_share'])})."
            )
        else:
            lines.append(
                f"Target `{name}` has no class at or above {IMBALANCE_SHARE:.0%}: "
                f"the largest is {profile['majority_level']} at "
                f"{profile['majority_count']}/{profile['n_non_null']} "
                f"({format_percent(profile['majority_share'])})."
            )
        return lines

    numeric = profile["numeric"]
    sentence = (
        f"Target `{name}` (n={profile['n_non_null']}): "
        f"mean {format_number(numeric['mean'])}, "
        f"standard deviation {format_number(numeric['std'])}, "
        f"skew {format_number(numeric['skew'])}, "
        f"minimum {format_number(numeric['min'])}, "
        f"maximum {format_number(numeric['max'])}."
    )
    if numeric["skew"] is not None and numeric["skew"] >= SKEW_FLAG:
        sentence += " The distribution is right-skewed."
    elif numeric["skew"] is not None and numeric["skew"] <= -SKEW_FLAG:
        sentence += " The distribution is left-skewed."
    lines.append(sentence)
    if profile["near_constant"]:
        lines.append(
            f"Target `{name}` is near-constant: {profile['majority_level']} is "
            f"{profile['majority_count']}/{profile['n_non_null']} non-null rows "
            f"({format_percent(profile['majority_share'])})."
        )
    return lines


def analyze(
    frame: pd.DataFrame,
    target: str,
    task: str = "auto",
    source: str | None = None,
) -> dict:
    """Compute the EDA summary for ``frame``. Column names come from the frame."""
    if not isinstance(frame, pd.DataFrame):
        raise EDAError("analyze expects a pandas DataFrame.")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise EDAError("Data file has no rows.")
    require_target(frame, target)
    decision = resolve_task(frame[target], task)
    overview = _overview(frame, target)
    numeric = []
    categorical = []
    for name in frame.columns:
        if name == target:
            continue
        series = frame[name]
        if is_numeric(series):
            profile = numeric_profile(series)
            profile["column"] = str(name)
            numeric.append(profile)
        else:
            profile = categorical_profile(series)
            profile["column"] = str(name)
            categorical.append(profile)
    result = {
        "source": source,
        "n_rows": int(frame.shape[0]),
        "n_cols": int(frame.shape[1]),
        "memory_bytes": overview["memory_bytes"],
        "target": str(target),
        "task": decision.task,
        "task_source": decision.source,
        "task_reason": decision.reason,
        "columns": overview["columns"],
        "preview_columns": overview["preview_columns"],
        "preview_rows": overview["preview_rows"],
        "preview_omitted_columns": overview["preview_omitted_columns"],
        "duplicates": _duplicates(frame),
        "missing": missingness(frame),
        "target_profile": target_profile(frame[target], decision.task),
        "numeric": numeric,
        "categorical": categorical,
        "associations": _associations(frame, target, decision.task),
        "correlation": _correlation(frame),
    }
    result["findings"] = _findings(result)
    return result
