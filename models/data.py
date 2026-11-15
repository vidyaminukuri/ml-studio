"""Load a CSV and make a stratified classification split.

Column names come from the file. Only the command line chooses the target.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from models import ModelError

MAX_CLASSES = 30
MIN_ROWS = 20


def is_numeric(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    )


def load_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise ModelError(f"Data file not found: {source}")
    if not source.is_file():
        raise ModelError(f"Data path is not a file: {source}")
    if source.stat().st_size == 0:
        raise ModelError(f"Data file is empty: {source}")
    try:
        frame = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise ModelError(f"Data file is empty: {source}") from exc
    except pd.errors.ParserError as exc:
        raise ModelError(f"Could not parse CSV {source}: {exc}") from exc
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ModelError(f"Data file has no rows: {source}")
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        names = ", ".join(repr(str(name)) for name in duplicated)
        raise ModelError(f"Data file has duplicate column names: {names}")
    return frame


def prepare_split(
    frame: pd.DataFrame,
    target: str,
    *,
    test_size: float,
    seed: int,
) -> dict:
    """Drop unusable rows and return a stratified train/test split."""
    if target is None or str(target).strip() == "":
        raise ModelError("A target column is required.")
    if not 0.1 <= float(test_size) <= 0.5:
        raise ModelError("--test-size must be between 0.1 and 0.5.")
    if target not in frame.columns:
        available = ", ".join(str(column) for column in frame.columns)
        raise ModelError(
            f"Target column {target!r} is not in the data. Available columns: {available}"
        )
    rows_loaded = int(frame.shape[0])
    missing_target = int(frame[target].isna().sum())
    work = frame.loc[frame[target].notna()].copy()
    duplicate_rows = int(work.duplicated().sum())
    work = work.drop_duplicates().reset_index(drop=True)
    if len(work) < MIN_ROWS:
        raise ModelError(
            f"Need at least {MIN_ROWS} rows after dropping missing targets and "
            f"exact duplicate rows; found {len(work)}."
        )
    labels = work[target]
    counts = labels.value_counts(dropna=True)
    if counts.size < 2:
        raise ModelError(
            f"Target {target!r} has fewer than 2 classes after cleaning, so it cannot be classified."
        )
    if counts.size > MAX_CLASSES:
        raise ModelError(
            f"Target {target!r} has {int(counts.size)} distinct values. "
            f"These applications classify a target with at most {MAX_CLASSES} classes."
        )
    if int(counts.min()) < 2:
        rare = ", ".join(str(level) for level, count in counts.items() if int(count) < 2)
        raise ModelError(
            "Every class needs at least 2 rows so the split can be stratified. "
            f"Classes with one row: {rare}."
        )
    feature_names = [column for column in work.columns if column != target]
    empty = [str(column) for column in feature_names if work[column].isna().all()]
    feature_names = [column for column in feature_names if column not in empty]
    if not feature_names:
        raise ModelError("No feature columns remain after dropping the target and empty columns.")
    numeric = [str(column) for column in feature_names if is_numeric(work[column])]
    categorical = [str(column) for column in feature_names if column not in numeric]
    features = work.loc[:, feature_names]
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=float(test_size),
            random_state=int(seed),
            stratify=labels,
        )
    except ValueError as exc:
        raise ModelError(
            "The stratified split failed. A class may be too small for this test size. "
            f"{exc}"
        ) from exc
    return {
        "source_rows": rows_loaded,
        "missing_target_rows": missing_target,
        "duplicate_rows_removed": duplicate_rows,
        "rows_used": int(len(work)),
        "empty_columns": empty,
        "numeric": numeric,
        "categorical": categorical,
        "class_counts": [(str(level), int(count)) for level, count in counts.items()],
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }
