import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, List, Literal

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData


class _HeatmapBiserial(_Plot):
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
        selected_columns: Optional[List[str]] = None,
        selected_value_columns: Optional[List[str]] = None,
        selected_missing_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.9,
        colorscale: str = "RdBu",
        show_values: bool = True,
        max_columns: int = 0,
        drop_constant_columns: bool = True,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 1,
        show_colorbar: bool = True,
        show_upper_triangle: bool = False,
        nan_color: str = "#c7c7c7",
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.selected_value_columns = selected_value_columns
        self.selected_missing_columns = selected_missing_columns
        self.ignore_high_missingness = ignore_high_missingness
        self.high_missingness_threshold = high_missingness_threshold
        self.colorscale = colorscale
        self.show_values = show_values
        self.max_columns = max_columns
        self.drop_constant_columns = drop_constant_columns
        self.order_by_missingness = order_by_missingness
        self.order = order
        self.value_round = value_round
        self.show_colorbar = show_colorbar
        self.show_upper_triangle = show_upper_triangle
        self.nan_color = nan_color

    # ------------------------------------------------------------------
    # Matrix preparation
    # ------------------------------------------------------------------

    def _get_correlation_matrix(self) -> pd.DataFrame:
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

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self) -> go.Figure:
        corr = self._get_correlation_matrix()

        effective_show_values = self.show_values and max(corr.shape) <= 30

        if self.show_upper_triangle and corr.shape[0] == corr.shape[1]:
            tri_mask = pd.DataFrame(
                np.tril(np.ones(corr.shape, dtype=bool), k=-1),
                index=corr.index,
                columns=corr.columns,
            )
            corr = corr.mask(tri_mask)

        x_labels = corr.columns.tolist()
        y_labels = corr.index.tolist()
        corr_values = corr.to_numpy(dtype=float)
        nan_mask = np.isnan(corr_values)

        if effective_show_values:
            text = np.empty(corr_values.shape, dtype=object)
            rounded = np.round(corr_values, self.value_round)
            for idx in np.ndindex(text.shape):
                raw = corr_values[idx]
                r = rounded[idx]
                if np.isnan(r) or np.isclose(r, 0.0):
                    text[idx] = ""
                elif np.isclose(abs(r), 1.0) and not np.isclose(abs(raw), 1.0):
                    text[idx] = "<1" if r > 0 else ">-1"
                else:
                    fmt = f"{r:.{self.value_round}f}"
                    text[idx] = fmt.rstrip("0").rstrip(".")
        else:
            text = None

        fig = go.Figure()

        if nan_mask.any():
            fig.add_trace(
                go.Heatmap(
                    z=np.where(nan_mask, 1.0, np.nan),
                    x=x_labels,
                    y=y_labels,
                    colorscale=[[0, self.nan_color], [1, self.nan_color]],
                    showscale=False,
                    hoverinfo="skip",
                )
            )

        colorbar = dict(
            title=(
                "Value/missingness correlation"
                "<br><span style='font-size:10px'>blue = higher → missing | red = higher → present</span>"
                "<br><span style='font-size:10px'>NaN = non-numeric or constant</span>"
            ),
            tickmode="array",
            tickvals=[-1, 0, 1],
            ticktext=["Higher values → present", "No association", "Higher values → missing"],
        ) if self.show_colorbar else None

        fig.add_trace(
            go.Heatmap(
                z=corr_values,
                x=x_labels,
                y=y_labels,
                colorscale=self.colorscale,
                zmin=-1,
                zmax=1,
                zmid=0,
                showscale=self.show_colorbar,
                text=text if effective_show_values else None,
                texttemplate="%{text}" if effective_show_values else None,
                colorbar=colorbar,
                hovertemplate=(
                    "<b>Value column</b>: %{y}<br>"
                    "<b>Missing column</b>: %{x}<br>"
                    "Association: %{z:.2f}<extra></extra>"
                ),
                hoverongaps=False,
            )
        )

        fig.update_xaxes(tickangle=-45, title_text="Missing column")
        fig.update_yaxes(tickangle=0, title_text="Value column", title_standoff=15)
        self._apply_base_layout(fig)
        return fig
