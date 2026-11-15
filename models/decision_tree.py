"""Decision tree classification application.

    python -m models.decision_tree --data PATH --target COLUMN --out DIR
"""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

MODEL_NAME = "decision_tree"
MODEL_NOTE = (
    "A single decision tree with max depth 6, at least 5 rows per leaf, "
    "and balanced class weights."
)


def build_estimator(seed: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
    )


def main(argv: list[str] | None = None) -> int:
    from models.run import run_application

    return run_application(MODEL_NAME, build_estimator, MODEL_NOTE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
