"""Gradient boosting classification application.

    python -m models.gradient_boosting --data PATH --target COLUMN --out DIR
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier

MODEL_NAME = "gradient_boosting"
MODEL_NOTE = (
    "Gradient boosting with 150 trees, learning rate 0.08, and depth 3. "
    "This estimator has no class-weight parameter."
)


def build_estimator(seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3,
        random_state=seed,
    )


def main(argv: list[str] | None = None) -> int:
    from models.run import run_application

    return run_application(MODEL_NAME, build_estimator, MODEL_NOTE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
