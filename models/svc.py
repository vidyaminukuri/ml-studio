"""Support-vector classification application.

    python -m models.svc --data PATH --target COLUMN --out DIR
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

MODEL_NAME = "svc"
MODEL_NOTE = (
    "RBF support-vector classifier with balanced class weights. "
    "Probabilities come from calibration on the training rows so the model can join the soft vote."
)


def build_estimator(seed: int) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=seed),
        method="sigmoid",
        cv=3,
        ensemble=False,
    )


def main(argv: list[str] | None = None) -> int:
    from models.run import run_application

    return run_application(MODEL_NAME, build_estimator, MODEL_NOTE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
