from __future__ import annotations

from functools import cached_property
import numpy as np
import pandas as pd


class _MissingDataCoreMixin:
    """
    Fundamental masks and basic DataFrame properties.

    Provides the two boolean masks that every other mixin depends on:
    * mask_missing: True where a value is NaN
    * mask_present: True where a value exists

    Also exposes dataset-level shape properties and overall missingness totals.
    All other mixins declare these as type-hints only and rely on MissingData
    inheriting this mixin to provide the actual implementations.
    """

    data: pd.DataFrame  # provided by MissingData.__init__

    # ------------------------------------------------------------------
    # Core masks
    # ------------------------------------------------------------------

    @cached_property
    def mask_missing(self) -> pd.DataFrame:
        """
        Boolean mask. True where a cell is missing (NaN).

        Same shape as the original DataFrame.
        """
        return self.data.isna()

    @cached_property
    def mask_present(self) -> pd.DataFrame:
        """
        Boolean mask. True where a cell has a real value.

        Exact inverse of mask_missing.
        """
        return ~self.mask_missing

    @cached_property
    def mask_observed(self) -> np.ndarray:
        """
        Binary uint8 matrix (0 = missing, 1 = present).

        Useful for numerical operations that need a numeric representation
        of the missingness structure (e.g. correlation, MCAR test).
        """
        return self.mask_present.to_numpy(np.uint8)

    # ------------------------------------------------------------------
    # Basic DataFrame properties
    # ------------------------------------------------------------------

    @property
    def columns(self) -> list[str]:
        """Column names of the underlying DataFrame."""
        return self.data.columns.tolist()

    @property
    def number_of_rows(self) -> int:
        """Total number of rows in the DataFrame."""
        return len(self.data)

    @property
    def number_of_columns(self) -> int:
        """Total number of columns in the DataFrame."""
        return self.data.shape[1]

    # ------------------------------------------------------------------
    # Dataset-level missingness totals
    # ------------------------------------------------------------------

    @cached_property
    def total_missing_rate(self) -> float:
        """
        Overall fraction of missing values across the entire dataset (0.0 to 1.0).

        Computed as the mean of the boolean mask over all cells.
        """
        return float(self.mask_missing.values.mean())

    @cached_property
    def total_missing_count(self) -> int:
        """Total number of missing cells in the dataset."""
        return int(self.mask_missing.values.sum())
