"""Write the markdown report and the plots that accompany it.

Every number in the report comes from ``analyze``. This module formats those
stats and draws charts from the same frame.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eda.analyze import (
    AUTO_MAX_DISTINCT,
    CORR_FLAG,
    HIGH_CARDINALITY,
    HIGH_CARDINALITY_MIN_ROWS,
    IMBALANCE_SHARE,
    IQR_K,
    MISSING_CHART_MAX,
    NEAR_CONSTANT_SHARE,
    PLOT_PAGE,
    PREVIEW_ROWS,
    RARE_LEVEL_SHARE,
    SCATTER_MAX,
    SKEW_FLAG,
    TOP_LEVELS,
    format_number,
    format_percent,
    label,
)

_COLOR = "#3D5A80"


def _method_text() -> str:
    return (
        f"Task detection (`auto`): a boolean, categorical, or string target is classification; "
        f"a numeric target is classification when it has at most {AUTO_MAX_DISTINCT} distinct "
        f"non-null values, and regression otherwise. A target with no non-null values follows "
        f"its storage type (numeric -> regression, otherwise classification). "
        f"An explicit task value replaces this rule. "
        f"Constant means fewer than 2 distinct non-null values. "
        f"Near-constant means the most common non-null value is at least {NEAR_CONSTANT_SHARE:.0%} "
        f"of non-null rows. Imbalanced, for classification, means the largest class is at least "
        f"{IMBALANCE_SHARE:.0%} and the target is not constant or near-constant. "
        f"Skew is the bias-corrected Fisher-Pearson coefficient; a column is flagged when "
        f"|skew| >= {SKEW_FLAG:g}. Quartiles are linear percentiles. Standard deviation uses the "
        f"sample divisor (n-1). An IQR outlier is a finite value outside "
        f"[Q1 - {IQR_K:g}*IQR, Q3 + {IQR_K:g}*IQR]. "
        f"A categorical level is rare when its share of non-null rows is at most {RARE_LEVEL_SHARE:.0%}. "
        f"High cardinality means at least {HIGH_CARDINALITY} distinct non-null levels, or a distinct "
        f"value on every non-null row when that row count is at least {HIGH_CARDINALITY_MIN_ROWS}. "
        f"A high correlation is a Pearson |r| >= {CORR_FLAG:g} on pairwise-complete rows. "
        f"The preview is the first {PREVIEW_ROWS} rows of the file, not a random sample. "
        f"Charts show at most {TOP_LEVELS} levels of a categorical column, at most {MISSING_CHART_MAX} "
        f"columns on the missingness chart, and at most {PLOT_PAGE} panels per figure. "
        f"Scatter plots sample down to {SCATTER_MAX} points when a pair is larger. "
        f"Tables and correlations still use every row."
    )


def _memory(n_bytes: int) -> str:
    if n_bytes < 1024:
        human = f"{n_bytes} B"
    elif n_bytes < 1024**2:
        human = f"{n_bytes / 1024:.1f} KB"
    else:
        human = f"{n_bytes / 1024**2:.1f} MB"
    return f"{human} ({n_bytes} bytes)"


def _cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    if len(text) > 48:
        return text[:45] + "..."
    return text


def _table(headers: list[str], rows: list[list[object]]) -> list[str]:
    if not rows:
        return ["_None._"]
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_cell(cell) for cell in row) + " |"
        for row in rows
    ]
    return [head, rule, *body]


def _style() -> None:
    sns.set_theme(style="whitegrid")
    matplotlib.rcParams["axes.titlesize"] = 11
    matplotlib.rcParams["axes.labelsize"] = 9
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _pages(items: list, page_size: int = PLOT_PAGE) -> list[list]:
    if not items:
        return []
    return [items[start : start + page_size] for start in range(0, len(items), page_size)]


def _page_name(stem: str, page: int, n_pages: int) -> str:
    if n_pages == 1:
        return f"{stem}.png"
    return f"{stem}_{page + 1}.png"


def _grid(n_panels: int, width: float = 4.3, height: float = 3.2) -> tuple[plt.Figure, np.ndarray]:
    cols = min(3, n_panels)
    rows = math.ceil(n_panels / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(width * cols, height * rows),
        squeeze=False,
    )
    return fig, axes


def _hide_unused(axes: np.ndarray, used: int) -> None:
    for ax in axes.ravel()[used:]:
        ax.set_visible(False)


def _prepare(out_dir: Path) -> Path:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for path in figures.glob("*"):
        if path.is_file():
            path.unlink()
    return figures


def _plot_missingness(result: dict, figures: Path) -> list[str]:
    columns = [row for row in result["missing"]["columns"] if row["count"] > 0]
    if not columns:
        return []
    columns.sort(key=lambda row: (-row["rate"], row["column"]))
    shown = list(reversed(columns[:MISSING_CHART_MAX]))
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.32 * len(shown) + 1.0)))
    ax.barh(
        [row["column"] for row in shown],
        [row["rate"] * 100 for row in shown],
        color=_COLOR,
    )
    ax.set_xlabel("Missing rows (%)")
    ax.set_title("Missingness by column")
    path = _save(fig, figures / "missingness.png")
    return [f"figures/{path.name}"]


def _missingness_truncated(result: dict) -> bool:
    missing = [row for row in result["missing"]["columns"] if row["count"] > 0]
    return len(missing) > MISSING_CHART_MAX


def _plot_target(frame: pd.DataFrame, result: dict, figures: Path) -> str | None:
    series = frame[result["target"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if result["task"] == "classification":
        classes = result["target_profile"]["classes"]
        if not classes:
            plt.close(fig)
            return None
        ax.bar(
            [row["level"] for row in classes[:30]],
            [row["count"] for row in classes[:30]],
            color=_COLOR,
        )
        ax.set_ylabel("Rows")
        ax.set_xlabel(result["target"])
        ax.set_title("Target class counts")
        ax.tick_params(axis="x", rotation=30)
    else:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        if values.size == 0:
            plt.close(fig)
            return None
        bins = 5 if np.unique(values).size == 1 else min(30, max(8, int(np.sqrt(values.size))))
        ax.hist(values, bins=bins, color=_COLOR, edgecolor="white")
        ax.set_xlabel(result["target"])
        ax.set_ylabel("Rows")
        ax.set_title("Target distribution")
    path = _save(fig, figures / "target.png")
    return f"figures/{path.name}"


def _numeric_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return values[np.isfinite(values)]


def _plot_numeric_histograms(frame: pd.DataFrame, result: dict, figures: Path) -> list[str]:
    columns = [column["column"] for column in result["numeric"]]
    written = []
    pages = _pages(columns)
    for page, group in enumerate(pages):
        fig, axes = _grid(len(group))
        for ax, name in zip(axes.ravel(), group, strict=False):
            values = _numeric_values(frame[name])
            if values.size == 0:
                ax.set_title(name)
                ax.text(0.5, 0.5, "no finite values", ha="center", va="center")
                ax.set_axis_off()
                continue
            bins = 5 if np.unique(values).size == 1 else min(30, max(8, int(np.sqrt(values.size))))
            ax.hist(values, bins=bins, color=_COLOR, edgecolor="white")
            ax.set_title(name)
            ax.set_ylabel("Rows")
        _hide_unused(axes, len(group))
        name = _page_name("numeric_histograms", page, len(pages))
        path = _save(fig, figures / name)
        written.append(f"figures/{path.name}")
    return written


def _plot_numeric_boxplots(frame: pd.DataFrame, result: dict, figures: Path) -> list[str]:
    columns = [column["column"] for column in result["numeric"]]
    written = []
    pages = _pages(columns)
    for page, group in enumerate(pages):
        fig, axes = _grid(len(group))
        for ax, name in zip(axes.ravel(), group, strict=False):
            values = _numeric_values(frame[name])
            if values.size == 0:
                ax.set_title(name)
                ax.text(0.5, 0.5, "no finite values", ha="center", va="center")
                ax.set_axis_off()
                continue
            sns.boxplot(y=values, ax=ax, color="#9BB8D3", width=0.45)
            ax.set_xlabel(name)
            ax.set_ylabel("")
            ax.set_title(name)
        _hide_unused(axes, len(group))
        name = _page_name("numeric_boxplots", page, len(pages))
        path = _save(fig, figures / name)
        written.append(f"figures/{path.name}")
    return written


def _plot_correlation(result: dict, figures: Path) -> str | None:
    correlation = result["correlation"]
    columns = correlation["columns"]
    if len(columns) < 2 or not correlation["matrix"]:
        return None
    matrix = pd.DataFrame(correlation["matrix"], index=columns, columns=columns)
    size = max(4.8, 0.55 * len(columns) + 2.2)
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=len(columns) <= 8,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Numeric correlation")
    path = _save(fig, figures / "correlation.png")
    return f"figures/{path.name}"


def _class_order(association: dict) -> list[str]:
    if not association["by_class"]:
        return []
    return [row["level"] for row in association["by_class"]]


def _plot_numeric_vs_target(
    frame: pd.DataFrame, result: dict, figures: Path
) -> list[str]:
    associations = result["associations"]["numeric"]
    if not associations:
        return []
    target = result["target"]
    written = []
    pages = _pages(associations)
    for page, group in enumerate(pages):
        fig, axes = _grid(len(group), width=4.6, height=3.4)
        for ax, association in zip(axes.ravel(), group, strict=False):
            name = association["column"]
            feature = pd.to_numeric(frame[name], errors="coerce")
            if result["task"] == "classification":
                plot_frame = pd.DataFrame({"level": frame[target], "value": feature})
                plot_frame = plot_frame.loc[plot_frame["level"].notna()].copy()
                plot_frame["level"] = plot_frame["level"].map(label)
                numeric_value = pd.to_numeric(plot_frame["value"], errors="coerce")
                keep = numeric_value.map(lambda value: bool(np.isfinite(value)))
                plot_frame = plot_frame.loc[keep].copy()
                order = _class_order(association)
                if plot_frame.empty:
                    ax.set_title(name)
                    ax.text(0.5, 0.5, "no finite values", ha="center", va="center")
                    ax.set_axis_off()
                    continue
                sns.boxplot(
                    data=plot_frame,
                    x="level",
                    y="value",
                    order=order,
                    ax=ax,
                    color="#9BB8D3",
                )
                ax.set_xlabel(target)
                ax.set_ylabel(name)
                ax.set_title(name)
                ax.tick_params(axis="x", rotation=30)
            else:
                target_values = pd.to_numeric(frame[target], errors="coerce")
                mask = np.isfinite(feature.to_numpy(dtype="float64")) & np.isfinite(
                    target_values.to_numpy(dtype="float64")
                )
                x = feature.to_numpy(dtype="float64")[mask]
                y = target_values.to_numpy(dtype="float64")[mask]
                sampled = x.size > SCATTER_MAX
                if sampled:
                    take = np.random.default_rng(0).choice(x.size, size=SCATTER_MAX, replace=False)
                    x = x[take]
                    y = y[take]
                ax.scatter(x, y, s=14, alpha=0.7, color=_COLOR, edgecolors="none")
                ax.set_xlabel(name)
                ax.set_ylabel(target)
                ax.tick_params(axis="x", labelsize=8)
                if x.size and float(np.max(np.abs(x))) >= 10000:
                    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
                title = name
                if association["pearson_r"] is not None:
                    title = f"{name} (r = {format_number(association['pearson_r'])})"
                if sampled:
                    title = f"{title}; plot n={SCATTER_MAX}"
                ax.set_title(title)
        _hide_unused(axes, len(group))
        filename = _page_name("numeric_vs_target", page, len(pages))
        path = _save(fig, figures / filename)
        written.append(f"figures/{path.name}")
    return written


def _plot_categorical_vs_target(result: dict, figures: Path) -> list[str]:
    associations = result["associations"]["categorical"]
    if not associations:
        return []
    target = result["target"]
    written = []
    pages = _pages(associations, page_size=6)
    for page, group in enumerate(pages):
        cols = min(2, len(group))
        rows = math.ceil(len(group) / cols)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(6.2 * cols, 3.6 * rows),
            squeeze=False,
        )
        for ax, association in zip(axes.ravel(), group, strict=False):
            rows_data = association["rows"]
            levels = [row["level"] for row in rows_data]
            if not rows_data:
                ax.set_title(association["column"])
                ax.text(0.5, 0.5, "no rows", ha="center", va="center")
                ax.set_axis_off()
                continue
            if result["task"] == "classification":
                class_names = []
                if rows_data and rows_data[0].get("classes"):
                    class_names = [item["level"] for item in rows_data[0]["classes"]]
                bottoms = np.zeros(len(levels))
                colors = sns.color_palette("colorblind", n_colors=max(1, len(class_names)))
                for index, class_name in enumerate(class_names):
                    heights = []
                    for row in rows_data:
                        match = next(
                            (item for item in row["classes"] if item["level"] == class_name),
                            None,
                        )
                        rate = 0.0 if match is None or match["rate"] is None else match["rate"] * 100
                        heights.append(rate)
                    ax.bar(
                        levels,
                        heights,
                        bottom=bottoms,
                        color=colors[index],
                        label=class_name,
                    )
                    bottoms = bottoms + np.asarray(heights)
                ax.set_ylabel(f"Share of level (%)")
                if class_names:
                    ax.legend(title=target, fontsize=8)
            else:
                means = [
                    0.0 if row["mean"] is None else row["mean"] for row in rows_data
                ]
                ax.bar(levels, means, color=_COLOR)
                ax.set_ylabel(f"Mean {target}")
            ax.set_title(association["column"])
            ax.tick_params(axis="x", rotation=40)
        _hide_unused(axes, len(group))
        filename = _page_name("categorical_vs_target", page, len(pages))
        path = _save(fig, figures / filename)
        written.append(f"figures/{path.name}")
    return written


def _image_lines(paths: list[str], caption: str) -> list[str]:
    if not paths:
        return []
    lines = [""]
    for path in paths:
        lines.append(f"![{caption}]({path})")
        lines.append("")
    lines.append(caption + ".")
    return lines


def _preview_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return format_number(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    return value


def _section_overview(result: dict) -> list[str]:
    source = result["source"] if result["source"] else "not recorded"
    lines = [
        "## 1. Shape, dtypes, memory, and sample rows",
        "",
        f"The table has {result['n_rows']} rows and {result['n_cols']} columns.",
        f"Source: `{source}`.",
        f"Pandas deep memory: {_memory(result['memory_bytes'])}.",
        "",
        "### Column types",
        "",
    ]
    rows = [
        [
            column["name"],
            column["dtype"],
            column["kind"],
            column["role"],
            column["non_null"],
            column["distinct"],
        ]
        for column in result["columns"]
    ]
    lines.extend(
        _table(["column", "dtype", "kind", "role", "non-null", "distinct"], rows)
    )
    lines.extend(["", "### Sample rows", ""])
    if result["preview_omitted_columns"]:
        omitted = ", ".join(f"`{name}`" for name in result["preview_omitted_columns"])
        lines.append(
            f"The preview shows the first {len(result['preview_columns'])} columns. "
            f"Omitted here: {omitted}."
        )
        lines.append("")
    body = [
        [_preview_cell(record[name]) for name in result["preview_columns"]]
        for record in result["preview_rows"]
    ]
    lines.extend(_table(result["preview_columns"], body))
    lines.append("")
    lines.append(
        f"The preview is the first {len(result['preview_rows'])} rows of the file."
    )
    return lines


def _section_duplicates(result: dict) -> list[str]:
    duplicates = result["duplicates"]
    return [
        "## 2. Duplicate rows",
        "",
        "A duplicate is an exact match on every column, including the target. "
        "Extra copies count rows after the first occurrence. "
        "Rows in a duplicated group count every row that shares its values with another row.",
        "",
        f"Extra copies: {duplicates['extra']} ({format_percent(duplicates['extra_rate'])} of {result['n_rows']} rows).",
        f"Rows in a duplicated group: {duplicates['in_groups']}.",
    ]


def _section_missing(result: dict, figure: list[str]) -> list[str]:
    missing = result["missing"]
    lines = [
        "## 3. Missingness",
        "",
        f"Missing cells: {missing['n_missing_cells']} of {missing['n_cells']} "
        f"({format_percent(missing['cell_rate'])}).",
        f"Rows with at least one missing value: {missing['rows_with_any']} "
        f"({format_percent(missing['rows_with_any_rate'])}).",
        f"Most missing cells in one row: {missing['max_missing_in_row']}.",
        "",
        "### By column",
        "",
    ]
    rows = [
        [row["column"], row["count"], format_percent(row["rate"])]
        for row in missing["columns"]
    ]
    lines.extend(_table(["column", "missing rows", "missing rate"], rows))
    lines.extend(["", "### By row", ""])
    histogram = [
        [row["n_missing"], row["rows"], format_percent(row["rate"])]
        for row in missing["row_histogram"]
    ]
    lines.extend(
        _table(["missing cells in the row", "rows", "share of rows"], histogram)
    )
    if figure:
        caption = "Share of rows missing, for columns that have a missing value"
        if _missingness_truncated(result):
            caption += f" (the {MISSING_CHART_MAX} columns with the highest missing rate)"
        lines.extend(_image_lines(figure, caption))
    else:
        lines.extend(["", "No missingness chart was written because no column has a missing value."])
    return lines


def _section_target(result: dict, figure: str | None) -> list[str]:
    profile = result["target_profile"]
    lines = [
        "## 4. Target profile",
        "",
        f"Target `{result['target']}`, task {result['task']}.",
        result["task_reason"],
        "",
        f"Non-null values: {profile['n_non_null']}. "
        f"Missing: {profile['missing']} ({format_percent(profile['missing_rate'])}). "
        f"Distinct non-null values: {profile['n_distinct']}.",
        "",
    ]
    if profile["constant"]:
        lines.append("The target is constant.")
    elif profile["near_constant"]:
        lines.append("The target is near-constant.")
    elif profile["imbalanced"]:
        lines.append("The target is imbalanced.")
    else:
        lines.append("The target is not constant, near-constant, or imbalanced under the cutoffs above.")
    lines.append("")
    if result["task"] == "classification":
        rows = [
            [row["level"], row["count"], format_percent(row["rate"])]
            for row in profile["classes"]
        ]
        lines.extend(_table(["class", "rows", "share of non-null"], rows))
    else:
        numeric = profile["numeric"]
        lines.extend(
            _table(
                ["count", "missing", "mean", "std", "min", "q1", "median", "q3", "max", "skew"],
                [[
                    numeric["count"],
                    numeric["missing"],
                    format_number(numeric["mean"]),
                    format_number(numeric["std"]),
                    format_number(numeric["min"]),
                    format_number(numeric["q1"]),
                    format_number(numeric["median"]),
                    format_number(numeric["q3"]),
                    format_number(numeric["max"]),
                    format_number(numeric["skew"]),
                ]],
            )
        )
    if figure:
        caption = "Target class counts" if result["task"] == "classification" else "Target distribution"
        lines.extend(_image_lines([figure], caption))
    elif profile["n_non_null"] == 0:
        lines.extend(["", "No target chart was written because the target has no non-null values."])
    return lines


def _section_numeric(result: dict, histograms: list[str], boxplots: list[str]) -> list[str]:
    lines = [
        "## 5. Numeric univariates",
        "",
        "Count is the number of finite values. Missing is the number of empty cells. "
        "Non-finite values, if any, are excluded from the summaries and counted separately. "
        "Outlier rate uses the finite count as the denominator.",
        "",
    ]
    if not result["numeric"]:
        lines.append("This table has no numeric features.")
        return lines
    show_non_finite = any(column["non_finite"] for column in result["numeric"])
    headers = [
        "column",
        "count",
        "missing",
        "mean",
        "std",
        "min",
        "q1",
        "median",
        "q3",
        "max",
        "skew",
        "lower fence",
        "upper fence",
        "IQR outliers",
        "outlier rate",
    ]
    if show_non_finite:
        headers.insert(3, "non-finite")
    rows = []
    for column in result["numeric"]:
        row = [
            column["column"],
            column["count"],
            column["missing"],
            format_number(column["mean"]),
            format_number(column["std"]),
            format_number(column["min"]),
            format_number(column["q1"]),
            format_number(column["median"]),
            format_number(column["q3"]),
            format_number(column["max"]),
            format_number(column["skew"]),
            format_number(column["lower_fence"]),
            format_number(column["upper_fence"]),
            column["outlier_count"],
            format_percent(column["outlier_rate"]),
        ]
        if show_non_finite:
            row.insert(3, column["non_finite"])
        rows.append(row)
    lines.extend(_table(headers, rows))
    lines.extend(_image_lines(histograms, "Histograms of the numeric features"))
    lines.extend(
        _image_lines(boxplots, "Boxplots of the numeric features; points beyond the whiskers are IQR outliers")
    )
    return lines


def _section_categorical(result: dict) -> list[str]:
    lines = [
        "## 6. Categorical univariates",
        "",
        "Cardinality and level shares use non-null values. "
        f"The frequency table shows the {TOP_LEVELS} most common levels. "
        f"A level is rare when its share of non-null rows is at most {RARE_LEVEL_SHARE:.0%}.",
        "",
    ]
    if not result["categorical"]:
        lines.append("This table has no categorical features.")
        return lines
    for column in result["categorical"]:
        lines.append(f"### `{column['column']}`")
        lines.append("")
        lines.append(
            f"Cardinality: {column['cardinality']}. "
            f"Non-null: {column['n_non_null']}. "
            f"Missing: {column['missing']} ({format_percent(column['missing_rate'])}). "
            f"High cardinality: {'yes' if column['high_cardinality'] else 'no'}."
        )
        lines.append("")
        lines.append("Top levels:")
        lines.append("")
        lines.extend(
            _table(
                ["level", "rows", "share of non-null"],
                [
                    [row["level"], row["count"], format_percent(row["rate"])]
                    for row in column["top"]
                ],
            )
        )
        lines.append("")
        if column["n_rare"] == 0:
            lines.append("Rare levels: none.")
        else:
            listed = column["rare"]
            lines.append(f"Rare levels: {column['n_rare']}.")
            lines.append("")
            lines.extend(
                _table(
                    ["level", "rows", "share of non-null"],
                    [
                        [row["level"], row["count"], format_percent(row["rate"])]
                        for row in listed
                    ],
                )
            )
            hidden = column["n_rare"] - len(listed)
            if hidden > 0:
                lines.append("")
                lines.append(f"{hidden} additional rare levels are not listed.")
        lines.append("")
    return lines


def _section_associations(result: dict, numeric_plots: list[str], categorical_plots: list[str]) -> list[str]:
    lines = [
        "## 7. Feature vs target",
        "",
    ]
    profile = result["target_profile"]
    if profile["n_non_null"] == 0:
        lines.append("Feature comparisons were skipped because the target has no non-null values.")
        return lines
    lines.append("Rows with a missing target are excluded from this section.")
    lines.append("")
    lines.append("### Numeric features")
    lines.append("")
    numeric = result["associations"]["numeric"]
    if not numeric:
        lines.append("This table has no numeric features.")
    elif result["task"] == "classification":
        lines.append("Each cell uses the finite values of that feature inside the class.")
        lines.append("")
        rows = []
        for association in numeric:
            for item in association["by_class"]:
                rows.append(
                    [
                        association["column"],
                        item["level"],
                        item["n"],
                        format_number(item["mean"]),
                        format_number(item["median"]),
                    ]
                )
        lines.extend(
            _table(["feature", "class", "finite n", "mean", "median"], rows)
        )
    else:
        lines.append("Pearson r uses pairwise-complete finite values.")
        lines.append("")
        rows = [
            [
                association["column"],
                association["n"],
                format_number(association["pearson_r"]),
            ]
            for association in numeric
        ]
        lines.extend(_table(["feature", "pairwise n", "pearson r"], rows))
    lines.extend(_image_lines(numeric_plots, "Numeric features against the target"))
    lines.extend(["", "### Categorical features", ""])
    categorical = result["associations"]["categorical"]
    if not categorical:
        lines.append("This table has no categorical features.")
    else:
        lines.append(
            f"Each table keeps the {TOP_LEVELS} largest levels. "
            "A `<missing>` level is kept as well when that feature has empty cells."
        )
        for association in categorical:
            lines.extend(["", f"#### `{association['column']}`", ""])
            if association["omitted_levels"]:
                lines.append(
                    f"{association['omitted_levels']} levels are omitted from this table."
                )
                lines.append("")
            if result["task"] == "classification":
                class_names = []
                if association["rows"] and association["rows"][0].get("classes"):
                    class_names = [item["level"] for item in association["rows"][0]["classes"]]
                headers = ["level", "n", *class_names]
                rows = []
                for row in association["rows"]:
                    rates = []
                    for class_name in class_names:
                        match = next(
                            (item for item in row["classes"] if item["level"] == class_name),
                            None,
                        )
                        rates.append("n/a" if match is None else format_percent(match["rate"]))
                    rows.append([row["level"], row["n"], *rates])
                lines.extend(_table(headers, rows))
                lines.append("")
                lines.append("Rates are the share of this level, among rows with a non-null target.")
            else:
                lines.extend(
                    _table(
                        ["level", "n", "mean target", "median target"],
                        [
                            [
                                row["level"],
                                row["n"],
                                format_number(row["mean"]),
                                format_number(row["median"]),
                            ]
                            for row in association["rows"]
                        ],
                    )
                )
    lines.extend(_image_lines(categorical_plots, "Categorical features against the target"))
    return lines


def _section_correlation(result: dict, figure: str | None) -> list[str]:
    correlation = result["correlation"]
    lines = [
        "## 8. Numeric correlation",
        "",
        "Pearson correlation on pairwise-complete rows. The matrix includes every numeric column, "
        "including a numeric target.",
        "",
    ]
    if len(correlation["columns"]) < 2:
        lines.append("Fewer than two numeric columns, so no correlation matrix was computed.")
        return lines
    if len(correlation["columns"]) <= 12:
        headers = ["column", *correlation["columns"]]
        rows = []
        for name, matrix_row in zip(correlation["columns"], correlation["matrix"], strict=True):
            rows.append([name, *[format_number(value) for value in matrix_row]])
        lines.extend(_table(headers, rows))
        lines.append("")
    else:
        lines.append(
            f"The matrix has {len(correlation['columns'])} numeric columns. "
            "It is shown in the figure rather than as a table."
        )
        lines.append("")
    if correlation["high_pairs"]:
        lines.append(f"Pairs with |r| >= {CORR_FLAG:g}:")
        lines.append("")
        lines.extend(
            _table(
                ["column", "column", "pearson r"],
                [
                    [pair["a"], pair["b"], format_number(pair["r"])]
                    for pair in correlation["high_pairs"]
                ],
            )
        )
    else:
        lines.append(f"No numeric pair has |r| >= {CORR_FLAG:g}.")
    if figure:
        lines.extend(_image_lines([figure], "Numeric correlation matrix"))
    return lines


def _section_findings(result: dict) -> list[str]:
    lines = ["## 9. Findings", ""]
    for line in result["findings"]:
        lines.append(f"- {line}")
    return lines


def write_report(frame: pd.DataFrame, result: dict, out_dir: str | Path) -> Path:
    """Write ``report.md`` and the figure files under ``out_dir``."""
    _style()
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figures = _prepare(destination)
    with warnings.catch_warnings():
        # Seaborn still calls a Matplotlib API that 3.11 has deprecated.
        warnings.filterwarnings("ignore", message="vert: bool was deprecated")
        warnings.filterwarnings("ignore", message="The set_bad function will be deprecated")
        missing_figure = _plot_missingness(result, figures)
        target_figure = _plot_target(frame, result, figures)
        histograms = _plot_numeric_histograms(frame, result, figures)
        boxplots = _plot_numeric_boxplots(frame, result, figures)
        correlation_figure = _plot_correlation(result, figures)
        numeric_vs_target = _plot_numeric_vs_target(frame, result, figures)
        categorical_vs_target = _plot_categorical_vs_target(result, figures)
    plt.close("all")

    lines = [
        "# Exploratory data analysis",
        "",
        f"Target: `{result['target']}`.",
        f"Task: {result['task']} ({result['task_source']}).",
        result["task_reason"],
        "",
        "## Method",
        "",
        _method_text(),
        "",
    ]
    lines.extend(_section_overview(result))
    lines.extend(["", *_section_duplicates(result)])
    lines.extend(["", *_section_missing(result, missing_figure)])
    lines.extend(["", *_section_target(result, target_figure)])
    lines.extend(["", *_section_numeric(result, histograms, boxplots)])
    lines.extend(["", *_section_categorical(result)])
    lines.extend(["", *_section_associations(result, numeric_vs_target, categorical_vs_target)])
    lines.extend(["", *_section_correlation(result, correlation_figure)])
    lines.extend(["", *_section_findings(result)])
    lines.append("")
    report_path = destination / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
