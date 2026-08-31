from typing import List, Optional

import pandas as pd

from missingfcup.core.missing_data import MissingData
from missingfcup.plots._association_heatmap import _AssociationHeatmap
from missingfcup.plots._selection import unusable_columns_error


class _HeatmapValue(_AssociationHeatmap):
    """
    Shared machinery for the two heatmaps that read a column's *values* against
    another column's missingness, which is the question MAR is about.

    Rows (y-axis): columns whose values are used as predictors.
    Columns (x-axis): columns whose missingness is being predicted.

    The matrix is asymmetric; no triangular masking is applied. Subclasses supply
    the statistic through ``_source()`` and say whether it is signed.
    """

    def _source(self) -> pd.DataFrame:
        raise NotImplementedError

    def __init__(
        self,
        data: MissingData,
        *,
        selected_value_columns: Optional[List[str]] = None,
        selected_missing_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)
        self.selected_value_columns = selected_value_columns
        self.selected_missing_columns = selected_missing_columns

    def _matrix(self) -> pd.DataFrame:
        corr = self._source().copy()

        if self.high_missingness_threshold is not None:
            miss_rate = self.data.col_missing_rate
            keep_rows = [
                c for c in corr.index if miss_rate.get(c, 0.0) < self.high_missingness_threshold
            ]
            keep_cols = [
                c for c in corr.columns if miss_rate.get(c, 0.0) < self.high_missingness_threshold
            ]
            corr = corr.loc[keep_rows, keep_cols]

        value_cols = self.selected_value_columns or self.selected_columns
        if value_cols is not None:
            value_cols = [c for c in value_cols if c in corr.index]
            if not value_cols:
                raise unusable_columns_error(
                    "selected_value_columns",
                    self.selected_value_columns or self.selected_columns,
                    corr.index,
                    self.data.columns,
                )
            corr = corr.loc[value_cols, :]

        missing_cols = self.selected_missing_columns or self.selected_columns
        if missing_cols is not None:
            missing_cols = [c for c in missing_cols if c in corr.columns]
            if not missing_cols:
                raise unusable_columns_error(
                    "selected_missing_columns",
                    self.selected_missing_columns or self.selected_columns,
                    corr.columns,
                    self.data.columns,
                )
            corr = corr.loc[:, missing_cols]

        if self.drop_constant_columns:
            # Drop value rows whose observed values are constant (all NaN in that row)
            all_nan_rows = corr.index[corr.isna().all(axis=1)]
            corr = corr.drop(index=all_nan_rows, errors="ignore")

            # Drop missing columns with no variance in missingness (always present/always missing)
            no_variance_cols = [
                c
                for c in corr.columns
                if c in self.data.mask_missing.columns and self.data.mask_missing[c].nunique() <= 1
            ]
            corr = corr.drop(columns=no_variance_cols, errors="ignore")

        if self.sort_by is not None and not corr.empty:
            # sort_by names what to order on, so it has to be read as a value rather
            # than tested for truth: treating it as a flag made "alphabetical" sort by
            # missing rate, silently, on this kind alone.
            if self.sort_by not in ("missingness", "alphabetical"):
                # The other kind gets this from select_columns, which the value
                # kinds do not use; without it an unknown value would sort by rate.
                raise ValueError(
                    f"sort_by must be 'missingness', 'alphabetical' or None, got {self.sort_by!r}."
                )
            if self.sort_by == "alphabetical":
                ordered_rows = sorted(corr.index, reverse=not self.ascending)
                ordered_cols = sorted(corr.columns, reverse=not self.ascending)
            else:
                miss_rate = self.data.col_missing_rate
                ordered_rows = (
                    miss_rate.loc[miss_rate.index.isin(corr.index)]
                    .sort_values(ascending=self.ascending)
                    .index
                )
                ordered_cols = (
                    miss_rate.loc[miss_rate.index.isin(corr.columns)]
                    .sort_values(ascending=self.ascending)
                    .index
                )
            corr = corr.loc[
                [c for c in ordered_rows if c in corr.index],
                [c for c in ordered_cols if c in corr.columns],
            ]

        if self.max_columns > 0:
            corr = corr.iloc[: self.max_columns, : self.max_columns]

        return corr

    def _axis_titles(self) -> tuple:
        return "Missing column", "Value column"


class _HeatmapDirection(_HeatmapValue):
    """
    Point-biserial correlation between a column's observed values and another
    column's missingness.

    Signed: the sign says which way the relationship runs, so it answers "do
    *higher* values of this column go with gaps in that one?". The cost of carrying
    a direction is that only relationships that have one are visible; a column whose
    gaps sit at both of its tails reads as zero here. Use the dependence kind for
    that case.
    """

    def _source(self) -> pd.DataFrame:
        return self.data.value_missing_corr

    def _axis_roles(self) -> tuple:
        return "values of", "missingness of", "Point-biserial association"


class _HeatmapDependence(_HeatmapValue):
    """
    Unsigned association between a column's observed values and another column's
    missingness, on a 0-1 scale from independence upwards.

    The statistic follows the value column's dtype: Kolmogorov-Smirnov for numeric
    columns, Cramer's V for categorical ones. Both measure distance from
    independence on the same scale, so one grid may hold both. It reports no
    direction, which is what buys it the ability to see relationships that have
    none.
    """

    def _source(self) -> pd.DataFrame:
        return self.data.value_missing_dependence

    def _is_signed(self) -> bool:
        return False

    def _axis_roles(self) -> tuple:
        return "values of", "missingness of", "Dependence"
