"""Classification applications for a generic table.

These modules are separate from the EDA package. Each model file can be run
on its own, and ``python -m models`` compares them and selects one.
"""

__all__ = ["ModelError"]


class ModelError(ValueError):
    """The table cannot be used for the requested classification run."""
