from __future__ import annotations

from functools import cached_property
from typing import Optional
import pandas as pd


class _MissingDataRowsMixin:
    """
    Row-level missingness metrics, row filtering, and row label helpers.

    All properties depend on mask_missing from _MissingDataCoreMixin,
    declared here as a type hint only.
    """

    mask_missing: pd.DataFrame
    data: pd.DataFrame

    # ------------------------------------------------------------------
    # Row metrics
    # ------------------------------------------------------------------

    @cached_property
    def row_missing_rate(self) -> pd.Series:
        """
        Proportion of missing values per row (0.0 to 1.0).

        1.0 = every value in the row is missing.
        0.0 = no missing values in the row.
        """
        return self.mask_missing.mean(axis=1)

    @cached_property
    def row_missing_count(self) -> pd.Series:
        """Number of missing values per row."""
        return self.mask_missing.sum(axis=1)

    @cached_property
    def row_missing_percent(self) -> pd.Series:
        """Percentage of missing values per row (0 to 100)."""
        return self.row_missing_rate * 100

    @cached_property
    def row_completeness(self) -> pd.Series:
        """
        Proportion of present (non-missing) values per row (0.0 to 1.0).

        Complement of row_missing_rate.
        """
        return 1 - self.row_missing_rate

    @cached_property
    def rows_complete(self) -> pd.Index:
        """Index labels of rows that contain no missing values."""
        return self.row_missing_count.loc[lambda s: s == 0].index

    @cached_property
    def rows_with_missing(self) -> pd.Index:
        """Index labels of rows that contain at least one missing value."""
        return self.row_missing_count.loc[lambda s: s > 0].index

    # ------------------------------------------------------------------
    # Row filtering
    # ------------------------------------------------------------------

    def rows_above_missing_threshold(self, threshold: float) -> pd.Index:
        """
        Return rows whose missing rate strictly exceeds ``threshold``.

        Parameters
        ----------
        threshold : float
            Value in [0, 1]. For example:
            * 0.2 → rows missing more than 20% of their values
            * 0.5 → rows missing more than half their values
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        return self.row_missing_rate.loc[lambda s: s > threshold].index

    # ------------------------------------------------------------------
    # Row label helper
    # ------------------------------------------------------------------

    def row_labels(self, index: Optional[pd.Index] = None) -> list[str]:
        """
        Convert row index values to plain strings for use as axis labels in plots.

        Parameters
        ----------
        index : pd.Index, optional
            The index to stringify. Pass a filtered/sorted df.index when a plot
            shows only a subset of rows. If None, uses the full DataFrame index.

        Examples
        --------
        DataFrame with index [0, 1, 2] → ['0', '1', '2']
        DataFrame with index ['A', 'B'] → ['A', 'B']
        """
        idx = self.data.index if index is None else index
        return [str(i) for i in idx]
