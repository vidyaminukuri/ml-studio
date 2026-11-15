"""Compare every classification application and keep the best one.

    python -m models.best_of --data PATH --target COLUMN --out DIR

The same comparison runs from ``python -m models``. Selection uses mean
cross-validated macro F1. Soft voting is one of the candidates.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from models.run import run_best

    return run_best(argv)


if __name__ == "__main__":
    raise SystemExit(main())
