"""Build the checked-in sample tables.

Only this module knows the sample schema. The analysis path never refers to
these column names.

Classification sample (``sample_classification.csv``), target ``churn``:

* ``age`` and ``tenure_years`` are numeric and strongly correlated.
* ``amount`` is right-skewed (lognormal).
* ``latency_ms`` is numeric with injected high outliers.
* ``score`` is numeric with missing values.
* ``region`` is a low-cardinality string.
* ``plan`` is a string with a rare level (``enterprise``).
* ``channel`` is a string with missing values.
* ``user_tag`` is a high-cardinality string.
* Five rows are exact copies of earlier rows.

Regression sample (``sample_regression.csv``), target ``spend`` (continuous,
so auto detection chooses regression):

* ``income`` is right-skewed and drives ``spend``.
* ``visits`` has injected outliers.
* ``rating`` is numeric with missing values.
* ``city`` is high-cardinality.
* ``segment`` has one rare level.
* ``referral`` has missing values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CLASSIFICATION_CSV = DATA_DIR / "sample_classification.csv"
REGRESSION_CSV = DATA_DIR / "sample_regression.csv"

CLASSIFICATION_TARGET = "churn"
REGRESSION_TARGET = "spend"
CLASSIFICATION_SEED = 42
REGRESSION_SEED = 7

COL_AGE = "age"
COL_TENURE = "tenure_years"
COL_AMOUNT = "amount"
COL_LATENCY = "latency_ms"
COL_SCORE = "score"
COL_REGION = "region"
COL_PLAN = "plan"
COL_CHANNEL = "channel"
COL_USER_TAG = "user_tag"
RARE_PLAN = "enterprise"

COL_INCOME = "income"
COL_VISITS = "visits"
COL_RATING = "rating"
COL_HOUSEHOLD = "household_size"
COL_CITY = "city"
COL_SEGMENT = "segment"
COL_REFERRAL = "referral"
RARE_SEGMENT = "government"


def classification_frame(seed: int = CLASSIFICATION_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 400
    age = np.clip(np.rint(rng.normal(42, 12, n)), 18, 78).astype(int)
    tenure = np.clip((age - 18) * 0.45 + rng.normal(0, 2.5, n), 0, None)
    tenure = np.round(tenure, 2)
    amount = np.round(rng.lognormal(mean=3.4, sigma=0.85, size=n), 2)
    latency = rng.normal(110, 18, n)
    outlier_at = rng.choice(n, size=10, replace=False)
    latency[outlier_at] = rng.uniform(900, 2200, size=outlier_at.size)
    latency = np.round(latency, 1)
    score = np.round(rng.normal(0.2, 1.0, n), 3)
    score[rng.choice(n, size=40, replace=False)] = np.nan

    region = rng.choice(["north", "south", "east", "west"], size=n, p=[0.41, 0.27, 0.20, 0.12])
    plan = rng.choice(["free", "pro", "team"], size=n, p=[0.70, 0.22, 0.08]).astype(object)
    plan[10] = RARE_PLAN
    plan[11] = RARE_PLAN
    channel = rng.choice(
        ["web", "ios", "android", "partner"], size=n, p=[0.50, 0.25, 0.20, 0.05]
    ).astype(object)
    channel[rng.choice(n, size=28, replace=False)] = None

    # Forty tags, each present at least five times, so the column is
    # high-cardinality without also being a pile of rare levels.
    n_tags = 40
    tag_ids = np.concatenate(
        [np.repeat(np.arange(n_tags), 5), rng.choice(n_tags, size=n - n_tags * 5)]
    )
    rng.shuffle(tag_ids)
    user_tag = np.array([f"tag_{int(i):03d}" for i in tag_ids])

    high_amount = amount >= np.median(amount)
    logit = (
        -1.85
        + 0.85 * (plan == "free")
        - 0.35 * (plan == "team")
        + 0.45 * (channel == "partner")
        + 0.35 * high_amount
        + 0.55 * (latency >= 500)
    )
    churn_p = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < churn_p, "yes", "no")

    frame = pd.DataFrame(
        {
            COL_AGE: age,
            COL_TENURE: tenure,
            COL_AMOUNT: amount,
            COL_LATENCY: latency,
            COL_SCORE: score,
            COL_REGION: region,
            COL_PLAN: plan,
            COL_CHANNEL: channel,
            COL_USER_TAG: user_tag,
            CLASSIFICATION_TARGET: churn,
        }
    )
    # Copy rows that are not the forced rare plan levels.
    copies = frame.iloc[390:395].copy()
    return pd.concat([frame, copies], ignore_index=True)


def regression_frame(seed: int = REGRESSION_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 160
    income = np.round(rng.lognormal(mean=10.6, sigma=0.55, size=n), 2)
    visits = rng.normal(12, 3, n)
    outlier_at = rng.choice(n, size=8, replace=False)
    visits[outlier_at] = rng.uniform(80, 160, size=outlier_at.size)
    visits = np.round(visits, 1)
    rating = np.round(rng.normal(3.6, 0.8, n), 3)
    rating[rng.choice(n, size=20, replace=False)] = np.nan
    household = rng.integers(1, 7, size=n)

    n_cities = 40
    city_ids = np.concatenate(
        [np.repeat(np.arange(n_cities), 2), rng.choice(n_cities, size=n - n_cities * 2)]
    )
    rng.shuffle(city_ids)
    city = np.array([f"city_{int(i):03d}" for i in city_ids])

    segment = rng.choice(["smb", "mid", "public"], size=n, p=[0.55, 0.30, 0.15]).astype(object)
    segment[4] = RARE_SEGMENT
    referral = rng.choice(["web", "email", "event"], size=n, p=[0.55, 0.30, 0.15]).astype(object)
    referral[rng.choice(n, size=12, replace=False)] = None

    log_income = np.log(income)
    standardized = (log_income - log_income.mean()) / log_income.std()
    spend = np.exp(3.1 + 0.55 * standardized + rng.normal(0, 0.35, n))
    spend = np.round(spend, 2)

    return pd.DataFrame(
        {
            COL_INCOME: income,
            COL_VISITS: visits,
            COL_RATING: rating,
            COL_HOUSEHOLD: household,
            COL_CITY: city,
            COL_SEGMENT: segment,
            COL_REFERRAL: referral,
            REGRESSION_TARGET: spend,
        }
    )


def write_sample_files(directory: Path | None = None) -> tuple[Path, Path]:
    destination = DATA_DIR if directory is None else Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    classification_path = destination / CLASSIFICATION_CSV.name
    regression_path = destination / REGRESSION_CSV.name
    classification_frame().to_csv(classification_path, index=False, float_format="%.6g")
    regression_frame().to_csv(regression_path, index=False, float_format="%.6g")
    return classification_path, regression_path


def main() -> None:
    classification_path, regression_path = write_sample_files()
    for path in (classification_path, regression_path):
        frame = pd.read_csv(path)
        print(f"{path}: {frame.shape[0]} rows, {frame.shape[1]} columns")


if __name__ == "__main__":
    main()
