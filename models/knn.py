"""k-nearest neighbors classification application.

    python -m models.knn --data PATH --target COLUMN --out DIR
"""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

MODEL_NAME = "knn"
MODEL_NOTE = (
    "k-nearest neighbors with k=7 and distance-weighted votes. "
    "Numeric features are standardized before distances are computed."
)


def build_estimator(seed: int) -> KNeighborsClassifier:
    del seed
    return KNeighborsClassifier(n_neighbors=7, weights="distance")


def main(argv: list[str] | None = None) -> int:
    from models.run import run_application

    return run_application(MODEL_NAME, build_estimator, MODEL_NOTE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
