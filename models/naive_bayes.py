"""Gaussian naive Bayes classification application.

    python -m models.naive_bayes --data PATH --target COLUMN --out DIR
"""

from __future__ import annotations

from sklearn.naive_bayes import GaussianNB

MODEL_NAME = "naive_bayes"
MODEL_NOTE = (
    "Gaussian naive Bayes on the standardized numeric columns and one-hot "
    "categorical columns. The model treats those transformed columns as independent."
)


def build_estimator(seed: int) -> GaussianNB:
    del seed
    return GaussianNB()


def main(argv: list[str] | None = None) -> int:
    from models.run import run_application

    return run_application(MODEL_NAME, build_estimator, MODEL_NOTE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
