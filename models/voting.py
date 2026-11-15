"""Soft voting ensemble.

    python -m models.voting --data PATH --target COLUMN --out DIR

The vote averages the class probabilities of every individual application.
"""

from __future__ import annotations

from sklearn.ensemble import VotingClassifier

MODEL_NAME = "voting"
MODEL_NOTE = (
    "Soft vote: the predicted class is the one with the highest average "
    "probability across logistic regression, the tree ensembles, the support-vector "
    "classifier, k-nearest neighbors, and naive Bayes. Each member has an equal vote."
)


def build_estimator(seed: int) -> VotingClassifier:
    from models.catalog import individual_models

    members = [(spec.name, spec.build(seed)) for spec in individual_models()]
    return VotingClassifier(estimators=members, voting="soft")


def main(argv: list[str] | None = None) -> int:
    from models.run import run_application

    return run_application(MODEL_NAME, build_estimator, MODEL_NOTE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
