"""AdaBoost classification application.

    python -m models.ada_boost --data PATH --target COLUMN --out DIR
"""

from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

MODEL_NAME = "ada_boost"
MODEL_NOTE = (
    "AdaBoost over 100 depth-2 trees with learning rate 0.8. "
    "This estimator has no class-weight parameter."
)


def build_estimator(seed: int) -> AdaBoostClassifier:
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=seed),
        n_estimators=100,
        learning_rate=0.8,
        random_state=seed,
    )


def main(argv: list[str] | None = None) -> int:
    from models.run import run_application

    return run_application(MODEL_NAME, build_estimator, MODEL_NOTE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
