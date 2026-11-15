"""Logistic regression classification application.

    python -m models.logistic_regression --data PATH --target COLUMN --out DIR
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

MODEL_NAME = "logistic_regression"
MODEL_NOTE = (
    "L2 logistic regression with balanced class weights. Numeric features are "
    "standardized, so a coefficient is the change per standard deviation."
)


def build_estimator(seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)


def main(argv: list[str] | None = None) -> int:
    from models.run import run_application

    return run_application(MODEL_NAME, build_estimator, MODEL_NOTE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
