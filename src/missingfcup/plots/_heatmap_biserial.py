import pandas as pd
from typing import Optional, List

from missingfcup.plots._association_heatmap import _AssociationHeatmap
from missingfcup.core.missing_data import MissingData


class _HeatmapBiserial(_AssociationHeatmap):
    """
    Heatmap of point-biserial correlations between column values and missingness indicators.

    Each cell [i, j] shows how strongly the observed values of column i associate
    with column j being missing, the key signal for MAR diagnosis.

    Rows (y-axis): columns whose values are used as predictors.
    Columns (x-axis): columns whose missingness is being predicted.

    The matrix is asymmetric; no triangular masking is applied.
    """

    def __init__(
        self,
        data: MissingData,
        selected_value_columns: Optional[List[str]] = None,
        selected_missing_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)
        self.selected_value_columns = selected_value_columns
        self.selected_missing_columns = selected_missing_columns

    def _matrix(self) -> pd.DataFrame:
        corr = self.data.value_missing_corr.copy()

        if self.ignore_high_missingness:
            miss_rate = self.data.col_missing_rate
            keep_rows = [c for c in corr.index if miss_rate.get(c, 0.0) < self.high_missingness_threshold]
            keep_cols = [c for c in corr.columns if miss_rate.get(c, 0.0) < self.high_missingness_threshold]
            corr = corr.loc[keep_rows, keep_cols]

        # Resolve value (row) columns
        value_cols = self.selected_value_columns or self.selected_columns
        if value_cols is not None:
            value_cols = [c for c in value_cols if c in corr.index]
            if not value_cols:
                raise ValueError("No selected_value_columns found in DataFrame.")
            corr = corr.loc[value_cols, :]

        # Resolve missing (column) columns
        missing_cols = self.selected_missing_columns or self.selected_columns
        if missing_cols is not None:
            missing_cols = [c for c in missing_cols if c in corr.columns]
            if not missing_cols:
                raise ValueError("No selected_missing_columns found in DataFrame.")
            corr = corr.loc[:, missing_cols]

        if self.drop_constant_columns:
            # Drop value rows whose observed values are constant (all NaN in that row)
            all_nan_rows = corr.index[corr.isna().all(axis=1)]
            corr = corr.drop(index=all_nan_rows, errors="ignore")

            # Drop missing columns with no variance in missingness (always present/always missing)
            no_variance_cols = [
                c for c in corr.columns
                if c in self.data.mask_missing.columns
                and self.data.mask_missing[c].nunique() <= 1
            ]
            corr = corr.drop(columns=no_variance_cols, errors="ignore")

        if self.order_by_missingness and not corr.empty:
            ascending = self.order == "asc"
            miss_rate = self.data.col_missing_rate

            ordered_rows = (
                miss_rate.loc[miss_rate.index.isin(corr.index)]
                .sort_values(ascending=ascending)
                .index
            )
            ordered_cols = (
                miss_rate.loc[miss_rate.index.isin(corr.columns)]
                .sort_values(ascending=ascending)
                .index
            )
            corr = corr.loc[
                [c for c in ordered_rows if c in corr.index],
                [c for c in ordered_cols if c in corr.columns],
            ]

        if self.max_columns > 0:
            corr = corr.iloc[: self.max_columns, : self.max_columns]

        return corr

    def _colorbar_config(self) -> dict:
        return dict(
            title=(
                "Value/missingness correlation"
                "<br><span style='font-size:10px'>blue = higher values missing | red = higher values present</span>"
                "<br><span style='font-size:10px'>NaN = non-numeric or constant</span>"
            ),
            tickmode="array",
            tickvals=[-1, 0, 1],
            ticktext=["Higher values present", "No association", "Higher values missing"],
        )

    def _hover_template(self) -> str:
        return (
            "<b>Value column</b>: %{y}<br>"
            "<b>Missing column</b>: %{x}<br>"
            "Association: %{z:.2f}<extra></extra>"
        )

    def _apply_axes(self, fig) -> None:
        fig.update_xaxes(tickangle=-45, title_text="Missing column")
        fig.update_yaxes(tickangle=0, title_text="Value column", title_standoff=15)
