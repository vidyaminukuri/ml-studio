"""Classification applications stay separate from EDA and write validation metrics."""

import pandas as pd
import pytest

from models import ModelError
from models.best_of import main as best_main
from models.data import load_frame, prepare_split
from models.logistic_regression import main as logistic_main
from models.run import select_winner


def test_logistic_regression_writes_validation_report(tmp_path):
    out = tmp_path / "logistic"
    code = logistic_main(
        [
            "--data",
            "eda/data/sample_classification.csv",
            "--target",
            "churn",
            "--out",
            str(out),
            "--cv",
            "3",
        ]
    )
    assert code == 0
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "accuracy" in text
    assert "F1, macro" in text
    assert "Permutation importance" in text
    assert "confusion" in text.lower()
    assert (out / "figures" / "confusion_matrix.png").is_file()
    assert (out / "figures" / "feature_importance.png").is_file()
    assert "0." in text


def test_best_of_selects_a_model_and_writes_the_leaderboard(tmp_path):
    out = tmp_path / "best"
    code = best_main(
        [
            "--data",
            "eda/data/sample_classification.csv",
            "--target",
            "churn",
            "--out",
            str(out),
            "--cv",
            "2",
        ]
    )
    assert code == 0
    text = (out / "report.md").read_text(encoding="utf-8")
    for name in [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "voting",
        "svc",
    ]:
        assert name in text
    assert "Selected model:" in text
    assert "holdout F1" in text
    assert (out / "figures" / "leaderboard.png").is_file()
    assert (out / "figures" / "feature_importance.png").is_file()


def test_select_winner_uses_cross_validated_macro_f1():
    def row(name, f1, auc=0.5):
        return {
            "model": name,
            "cv": {
                "f1_macro": {"mean": f1, "std": 0.01},
                "roc_auc_ovr": {"mean": auc, "std": 0.01},
            },
        }

    winner, reason = select_winner(
        [row("voting", 0.61), row("random_forest", 0.70, 0.8), row("knn", 0.70, 0.7)]
    )
    assert winner == "random_forest"
    assert "random_forest" in reason
    assert "holdout was not used" in reason


def test_rejects_empty_file_and_missing_target(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ModelError, match="empty"):
        load_frame(empty)
    assert logistic_main(["--data", str(empty), "--target", "y", "--out", str(tmp_path / "o")]) == 2

    missing = tmp_path / "table.csv"
    missing.write_text("a,b\n1,yes\n2,no\n", encoding="utf-8")
    with pytest.raises(ModelError, match="not in the data"):
        prepare_split(load_frame(missing), "churn", test_size=0.25, seed=1)
    assert (
        logistic_main(["--data", str(missing), "--target", "churn", "--out", str(tmp_path / "m")])
        == 2
    )


def test_single_class_target_fails(tmp_path):
    path = tmp_path / "one.csv"
    frame = pd.DataFrame({"x": range(25), "y": ["a"] * 25})
    frame.to_csv(path, index=False)
    with pytest.raises(ModelError, match="fewer than 2 classes"):
        prepare_split(load_frame(path), "y", test_size=0.25, seed=1)


def test_prepare_split_is_stratified_and_drops_duplicate_rows():
    frame = load_frame("eda/data/sample_classification.csv")
    bundle = prepare_split(frame, "churn", test_size=0.25, seed=42)
    assert bundle["duplicate_rows_removed"] >= 5
    assert bundle["rows_used"] == len(frame) - bundle["duplicate_rows_removed"]
    train_rate = (bundle["y_train"] == "yes").mean()
    test_rate = (bundle["y_test"] == "yes").mean()
    assert abs(train_rate - test_rate) < 0.05
    assert "amount" in bundle["numeric"]
    assert "plan" in bundle["categorical"]
