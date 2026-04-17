from __future__ import annotations

from functools import cached_property
import pandas as pd


class _MissingDataColumnsMixin:
    """
    Column-level missingness metrics and column filtering.

    All properties depend on mask_missing and number_of_rows from
    _MissingDataCoreMixin, declared here as type hints only.
    """

    mask_missing: pd.DataFrame
    number_of_rows: int

    # ------------------------------------------------------------------
    # Column metrics
    # ------------------------------------------------------------------

    @cached_property
    def col_missing_rate(self) -> pd.Series:
        """
        Proportion of missing values per column (0.0–1.0).

        1.0 = every value in the column is missing.
        0.0 = no missing values in the column.
        """
        return self.mask_missing.mean()

    @cached_property
    def col_missing_count(self) -> pd.Series:
        """Number of missing values per column."""
        return self.mask_missing.sum()

    @cached_property
    def col_present_count(self) -> pd.Series:
        """Number of observed (non-missing) values per column."""
        return self.number_of_rows - self.col_missing_count

    @cached_property
    def col_missing_percent(self) -> pd.Series:
        """Percentage of missing values per column (0–100)."""
        return self.col_missing_rate * 100

    @cached_property
    def col_completeness(self) -> pd.Series:
        """
        Proportion of present (non-missing) values per column (0.0–1.0).

        Complement of col_missing_rate.
        """
        return 1 - self.col_missing_rate

    @cached_property
    def cols_complete(self) -> pd.Index:
        """Column labels with zero missing values."""
        return self.col_missing_count.loc[lambda s: s == 0].index

    # ------------------------------------------------------------------
    # Column filtering
    # ------------------------------------------------------------------

    def columns_above_missing_threshold(self, threshold: float) -> pd.Index:
        """
        Return columns whose missing rate strictly exceeds ``threshold``.

        Parameters
        ----------
        threshold : float
            Value in [0, 1]. For example:
            - 0.2 → columns missing more than 20% of their values
            - 0.5 → columns missing more than half their values
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        return self.col_missing_rate.loc[lambda s: s > threshold].index
