"""Train-only preprocessing shared by every classification application."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models import ModelError


def make_pipeline(estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    """Impute and encode inside the pipeline so the fit uses training rows only."""
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric),
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical),
            )
        )
    if not transformers:
        raise ModelError("No usable feature columns.")
    prepare = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("prepare", prepare), ("model", estimator)])
