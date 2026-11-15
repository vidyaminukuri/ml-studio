# ml-studio

`index.html` is a small in-browser lab for tabular data. The `eda` package is a reusable exploratory-data-analysis report for a CSV: any mix of numeric and string columns, plus one target.

## Install

From the repository root:

```bash
pip install -r requirements.txt
```

## Run on your CSV

```bash
python -m eda --data PATH --target COLUMN --out DIR
python -m eda --data PATH --target COLUMN --task regression --out DIR
python -m eda --data PATH --target COLUMN --task classification --out DIR
```

`--task` is `auto`, `regression`, or `classification`. The report is `DIR/report.md`, with plots in `DIR/figures/`.

From Python:

```python
from eda import run_analysis

run_analysis("data.csv", target="y", task="auto", out="eda_report")
```

The analysis path does not assume column names. Only the sample generator knows the sample schema.

## Sample data

`python -m eda` with no `--data` and no `--target` runs the built-in classification sample and writes `./eda_report` (that directory is gitignored).

```bash
python -m eda --data eda/data/sample_classification.csv --target churn --out eda/examples/sample_report
python -m eda --data eda/data/sample_regression.csv --target spend --out eda_report_regression
```

The checked-in classification report is [eda/examples/sample_report/report.md](eda/examples/sample_report/report.md).

Regenerate the CSVs with `python -m eda.sample`. Nothing is downloaded.

The classification sample (`churn`, yes/no) has:

| column | role in the sample |
| --- | --- |
| `amount` | right-skewed numeric |
| `latency_ms` | numeric with high outliers |
| `score` | numeric with missing values |
| `age`, `tenure_years` | numeric pair with a strong correlation |
| `user_tag` | high-cardinality string |
| `plan` | string with a rare level (`enterprise`) |
| `channel` | string with missing values |
| `region` | low-cardinality string |

The regression sample uses a continuous `spend` target, so `--task auto` selects regression. `income` is skewed, `visits` has outliers, `rating` is missing in places, `city` is high-cardinality, `segment` has a rare level, and `referral` has missing values.

## How `--task auto` chooses

1. A target with no non-null values follows its storage type: numeric becomes regression, and anything else becomes classification.
2. A boolean, categorical, or string target is classification. Strings of digits stay classification; detection uses the dtype, not a guess that the text is numeric.
3. A numeric target with at most 10 distinct non-null values is classification. Missing values are not a level.
4. Otherwise the target is regression.

A numeric feature with few distinct values stays on the numeric path. The cutoff applies only to the target. `--task regression` or `--task classification` overrides the rule.

The report also flags a constant target (fewer than 2 distinct non-null values) and a near-constant target (the most common non-null value is at least 95% of non-null rows). For classification, a largest class of at least 70% is called imbalanced when the target is not already constant or near-constant.

## What the report contains

1. Shape, dtypes, memory, and the first rows of the file.
2. Exact duplicate rows.
3. Missingness by column and by row (counts and rates).
4. Target profile: class balance, or the distribution and skew, plus the constant / near-constant flag.
5. Numeric univariates (count, missing, mean, standard deviation, min, quartiles, max, skew), a histogram, and IQR outlier fences.
6. Categorical univariates: cardinality, top frequencies, rare levels, missing.
7. Feature vs target: class-wise numeric distributions, or Pearson correlation for regression; categorical levels show the target rate or the target mean.
8. Numeric Pearson correlation matrix and the high-correlation pairs.
9. Findings written from those stats.

Quartiles are linear percentiles. Standard deviation uses the sample divisor (n-1). Skew is the bias-corrected Fisher-Pearson coefficient. An IQR outlier sits outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`. A level is rare at or below 1% of non-null rows. High cardinality means 20 or more levels, or a different value on every non-null row when there are at least 10. A high correlation is `|r| >= 0.7`.

## Classification models

Prediction lives in `models/` and is separate from the EDA package. Each algorithm is its own application. `python -m models` fits all of them, including a soft voting ensemble, and selects the one with the highest mean cross-validated macro F1.

The local folder `C:\Convergence\PredictiveModels\ConvergenceMLHub\Current_Usage_Models` was not available here. The applications follow that description: standard classifiers, tree ensembles, a soft vote across them, and a best-of selector.

```bash
python -m models --data PATH --target COLUMN --out DIR
python -m models.logistic_regression --data PATH --target COLUMN --out DIR
python -m models.decision_tree --data PATH --target COLUMN --out DIR
python -m models.random_forest --data PATH --target COLUMN --out DIR
python -m models.extra_trees --data PATH --target COLUMN --out DIR
python -m models.gradient_boosting --data PATH --target COLUMN --out DIR
python -m models.ada_boost --data PATH --target COLUMN --out DIR
python -m models.svc --data PATH --target COLUMN --out DIR
python -m models.knn --data PATH --target COLUMN --out DIR
python -m models.naive_bayes --data PATH --target COLUMN --out DIR
python -m models.voting --data PATH --target COLUMN --out DIR
python -m models.best_of --data PATH --target COLUMN --out DIR
```

With no `--data` and no `--target`, the command uses `eda/data/sample_classification.csv` and target `churn`. A checked-in comparison is [models/examples/sample_report/report.md](models/examples/sample_report/report.md).

The report includes cross-validated and holdout accuracy, precision, recall, F1 (macro and weighted), ROC-AUC, log loss, a confusion matrix, a majority-class baseline, and permutation importance on the original columns. Tree and logistic models also list their internal weights. Hyperparameters are fixed, not searched. The holdout is not used to pick the winner.

## Tests

```bash
python -m pytest tests
```

## Limits

- The file must be a UTF-8 CSV, and the whole table is loaded into memory.
- The command fails if the file is empty, has a header but no rows, or does not contain `--target`.
- Duplicate detection is an exact match on every column after parsing.
- Correlation in the EDA report is Pearson, on pairwise-complete rows. That report does not fit a model. Classification fits live in `models/`.
- The preview is the first 5 rows, not a random sample, and it shows at most 12 columns.
- Charts show at most 10 levels of a categorical column, at most 40 columns on the missingness chart, and at most 9 panels per figure. Scatter plots draw at most 5000 points. The tables still use every row.
- Empty cells are missing. The EDA report does not impute them. Classification imputes inside the training pipeline only.
- Classification drops rows with a missing target and exact duplicate rows before the split. It needs at least 20 remaining rows, 2 to 30 classes, and at least 2 rows in every class.
- Model hyperparameters are fixed. The winner is the highest mean cross-validated macro F1. On the classification sample, that model can still lose to a majority-class baseline on holdout accuracy.
