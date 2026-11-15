"""Exploratory data analysis for a generic table with one target column."""

from eda.analyze import EDAError, analyze, detect_task, load_table, missingness
from eda.report import write_report

__all__ = [
    "EDAError",
    "analyze",
    "detect_task",
    "load_table",
    "missingness",
    "run_analysis",
    "write_report",
]


def run_analysis(
    data: str,
    target: str,
    task: str = "auto",
    out: str = "eda_report",
):
    """Load a CSV, summarize it, and write a markdown report plus plots."""
    frame = load_table(data)
    result = analyze(frame, target, task=task, source=str(data))
    return write_report(frame, result, out)
