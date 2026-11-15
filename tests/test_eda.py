"""Tests for task detection, missingness, and the sample report."""

import numpy as np
import pandas as pd
import pytest

from eda.analyze import (
    EDAError,
    analyze,
    detect_task,
    load_table,
    missingness,
    numeric_profile,
    resolve_task,
)
from eda.__main__ import main
from eda.sample import (
    CLASSIFICATION_CSV,
    CLASSIFICATION_TARGET,
    COL_AGE,
    COL_AMOUNT,
    COL_CHANNEL,
    COL_CITY,
    COL_INCOME,
    COL_LATENCY,
    COL_PLAN,
    COL_RATING,
    COL_REFERRAL,
    COL_SCORE,
    COL_SEGMENT,
    COL_TENURE,
    COL_USER_TAG,
    COL_VISITS,
    RARE_PLAN,
    RARE_SEGMENT,
    REGRESSION_CSV,
    REGRESSION_TARGET,
)

FIGURES = [
    "missingness.png",
    "target.png",
    "numeric_histograms.png",
    "numeric_boxplots.png",
    "correlation.png",
    "numeric_vs_target.png",
    "categorical_vs_target.png",
]


def test_task_detection():
    assert detect_task(pd.Series(["a", "b", "a"])).task == "classification"
    assert "not numeric" in detect_task(pd.Series(["0", "1", "2"])).reason
    assert detect_task(pd.Series([True, False, True])).task == "classification"

    ten = pd.Series(list(range(10)) + [None])
    eleven = pd.Series(list(range(11)) + [None])
    ten_decision = detect_task(ten)
    eleven_decision = detect_task(eleven)
    assert ten_decision.task == "classification"
    assert ten_decision.source == "auto"
    assert "at most 10" in ten_decision.reason
    assert eleven_decision.task == "regression"
    assert "more than 10" in eleven_decision.reason

    assert detect_task(pd.Series([np.nan, np.nan])).task == "regression"
    assert detect_task(pd.Series([None, None], dtype=object)).task == "classification"

    forced = resolve_task(eleven, "classification")
    assert forced.task == "classification"
    assert forced.source == "explicit"
    with pytest.raises(EDAError):
        resolve_task(eleven, "clustering")


def test_target_flags():
    near = pd.DataFrame({"x": range(100), "t": ["a"] * 96 + ["b"] * 4})
    near_result = analyze(near, "t")
    assert near_result["target_profile"]["near_constant"] is True
    assert near_result["target_profile"]["imbalanced"] is False
    assert any("near-constant" in line for line in near_result["findings"])

    imbalanced = pd.DataFrame({"x": range(10), "t": ["a"] * 7 + ["b"] * 3})
    imbalanced_result = analyze(imbalanced, "t")
    assert imbalanced_result["target_profile"]["imbalanced"] is True
    assert imbalanced_result["target_profile"]["near_constant"] is False
    assert any("imbalanced" in line for line in imbalanced_result["findings"])

    constant = pd.DataFrame({"x": [1.0, 2.0, 3.0], "t": ["a", "a", "a"]})
    constant_result = analyze(constant, "t")
    assert constant_result["target_profile"]["constant"] is True
    assert any("constant" in line for line in constant_result["findings"])


def test_missingness_and_findings():
    frame = pd.DataFrame(
        {
            "x": [1.0, None, 3.0, None],
            "y": ["a", "b", None, "a"],
            "t": ["no", "yes", "no", "no"],
        }
    )
    stats = missingness(frame)
    assert [row["count"] for row in stats["columns"]] == [2, 1, 0]
    assert [row["rate"] for row in stats["columns"]] == [0.5, 0.25, 0.0]
    assert stats["n_missing_cells"] == 3
    assert stats["n_cells"] == 12
    assert stats["rows_with_any"] == 3
    assert stats["max_missing_in_row"] == 1
    histogram = {row["n_missing"]: row["rows"] for row in stats["row_histogram"]}
    assert histogram == {0: 1, 1: 3}
    assert sum(row["count"] for row in stats["columns"]) == stats["n_missing_cells"]

    result = analyze(frame, "t")
    text = " ".join(result["findings"])
    assert "x (2/4, 50.0%)" in text
    assert "y (1/4, 25.0%)" in text
    assert "3 of 4 rows have at least one missing value" in text


def test_numeric_profile_rules():
    series = pd.Series([1.0, 2, 3, 4, 10, np.nan, 8])
    profile = numeric_profile(series)
    assert profile["skew"] == pytest.approx(series.skew())
    described = series.dropna().describe()
    assert profile["count"] == 6
    assert profile["missing"] == 1
    assert profile["mean"] == pytest.approx(described["mean"])
    assert profile["std"] == pytest.approx(described["std"])
    assert profile["min"] == pytest.approx(described["min"])
    assert profile["q1"] == pytest.approx(described["25%"])
    assert profile["median"] == pytest.approx(described["50%"])
    assert profile["q3"] == pytest.approx(described["75%"])
    assert profile["max"] == pytest.approx(described["max"])

    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=float)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    expected = int(np.sum((values < lower) | (values > upper)))
    outlier_profile = numeric_profile(pd.Series(values))
    assert expected >= 1
    assert outlier_profile["outlier_count"] == expected
    assert outlier_profile["lower_fence"] == pytest.approx(lower)
    assert outlier_profile["upper_fence"] == pytest.approx(upper)

    constant = numeric_profile(pd.Series([2.0, 2.0, 2.0, 2.0]))
    assert constant["skew"] is None
    assert constant["outlier_count"] == 0


def test_rejects_empty_file_and_missing_target(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(EDAError, match="empty"):
        load_table(empty)
    assert main(["--data", str(empty), "--target", "y", "--out", str(tmp_path / "out")]) == 2

    header_only = tmp_path / "header.csv"
    header_only.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(EDAError, match="no rows"):
        load_table(header_only)

    missing_target = tmp_path / "table.csv"
    missing_target.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(EDAError, match="not in the data"):
        analyze(load_table(missing_target), "y")
    assert (
        main(["--data", str(missing_target), "--target", "y", "--out", str(tmp_path / "missing")])
        == 2
    )

    absent = tmp_path / "absent.csv"
    with pytest.raises(EDAError, match="not found"):
        load_table(absent)

    assert main(["--data", str(missing_target), "--out", str(tmp_path / "half")]) == 2
    assert main(["--target", "y", "--out", str(tmp_path / "half2")]) == 2


def test_classification_sample_shape():
    frame = load_table(CLASSIFICATION_CSV)
    assert CLASSIFICATION_TARGET in frame.columns
    assert detect_task(frame[CLASSIFICATION_TARGET]).task == "classification"
    assert frame[COL_SCORE].isna().any()
    assert frame[COL_CHANNEL].isna().any()
    assert frame[COL_USER_TAG].nunique(dropna=True) >= 20
    assert numeric_profile(frame[COL_AMOUNT])["skew"] >= 1
    assert numeric_profile(frame[COL_LATENCY])["outlier_count"] > 0
    plan_share = (frame[COL_PLAN] == RARE_PLAN).sum() / frame[COL_PLAN].notna().sum()
    assert plan_share <= 0.01
    correlation = frame[[COL_AGE, COL_TENURE]].corr().iloc[0, 1]
    assert correlation >= 0.7
    assert frame.duplicated().sum() >= 5
    assert set(frame[CLASSIFICATION_TARGET].dropna().unique()) == {"yes", "no"}


def test_sample_run_writes_report(tmp_path):
    out = tmp_path / "classification"
    assert main(["--data", str(CLASSIFICATION_CSV), "--target", CLASSIFICATION_TARGET, "--out", str(out)]) == 0
    report = out / "report.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    frame = load_table(CLASSIFICATION_CSV)
    result = analyze(frame, CLASSIFICATION_TARGET, source=str(CLASSIFICATION_CSV))
    assert result["task"] == "classification"
    for line in result["findings"]:
        assert line in text
    assert "## 9. Findings" in text
    assert "1. Shape" in text
    assert "3. Missingness" in text
    assert "8. Numeric correlation" in text
    for name in FIGURES:
        assert (out / "figures" / name).is_file()

    default_out = tmp_path / "default"
    assert main(["--out", str(default_out)]) == 0
    assert (default_out / "report.md").is_file()
    assert CLASSIFICATION_TARGET in (default_out / "report.md").read_text(encoding="utf-8")


def test_regression_sample_run(tmp_path):
    frame = load_table(REGRESSION_CSV)
    decision = detect_task(frame[REGRESSION_TARGET])
    assert decision.task == "regression"
    assert "more than 10" in decision.reason
    assert frame[REGRESSION_TARGET].nunique(dropna=True) > 10
    assert numeric_profile(frame[COL_INCOME])["skew"] >= 1
    assert numeric_profile(frame[COL_VISITS])["outlier_count"] > 0
    assert frame[COL_RATING].isna().any()
    assert frame[COL_CITY].nunique(dropna=True) >= 20
    assert frame[COL_REFERRAL].isna().any()
    segment_share = (frame[COL_SEGMENT] == RARE_SEGMENT).sum() / frame[COL_SEGMENT].notna().sum()
    assert segment_share <= 0.01

    out = tmp_path / "regression"
    assert (
        main(["--data", str(REGRESSION_CSV), "--target", REGRESSION_TARGET, "--out", str(out)])
        == 0
    )
    text = (out / "report.md").read_text(encoding="utf-8")
    result = analyze(frame, REGRESSION_TARGET, source=str(REGRESSION_CSV))
    assert result["task"] == "regression"
    for line in result["findings"]:
        assert line in text
    assert (out / "figures" / "target.png").is_file()
    assert (out / "figures" / "correlation.png").is_file()
    assert "pearson r" in text
