"""Extra-trees classification application.

    python -m models.extra_trees --data PATH --target COLUMN --out DIR
"""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier

MODEL_NAME = "extra_trees"
MODEL_NOTE = (
    "Extra-trees ensemble of 300 trees. Each split uses a random threshold, "
    "with balanced class weights and at least 2 rows in each leaf."
)


def build_estimator(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def main(argv: list[str] | None = None) -> int:
    from models.run import run_application

    return run_application(MODEL_NAME, build_estimator, MODEL_NOTE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
