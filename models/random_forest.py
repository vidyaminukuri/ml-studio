"""Random forest classification application.

    python -m models.random_forest --data PATH --target COLUMN --out DIR
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

MODEL_NAME = "random_forest"
MODEL_NOTE = (
    "Random forest of 300 trees with balanced bootstrap samples and at least "
    "2 rows in each leaf."
)


def build_estimator(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )


def main(argv: list[str] | None = None) -> int:
    from models.run import run_application

    return run_application(MODEL_NAME, build_estimator, MODEL_NOTE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
