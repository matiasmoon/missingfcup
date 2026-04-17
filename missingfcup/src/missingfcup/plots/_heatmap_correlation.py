import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, List, Literal

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData


class _HeatmapCorrelation(_Plot):
    """
    Heatmap of column-level correlations through the lens of missingness.

    Parameters
    ----------
    mode : str
        Which correlation to display:
        - ``"missing_missing"``: columns that tend to miss in the same rows.
        - ``"present_present"``: columns that tend to be observed in the same rows.
        - ``"present_missing"``: being observed in one column vs. missing in another.
    """

    def __init__(
        self,
        data: MissingData,
        mode: Literal["missing_missing", "present_present", "present_missing"] = "missing_missing",
        selected_columns: Optional[List[str]] = None,
        colorscale: str = "RdBu",
        show_values: bool = True,
        max_columns: int = 0,  # 0 = show all variables by default
        drop_constant_columns: bool = True,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 1,
        show_colorbar: bool = True,
        show_upper_triangle: bool = False,
        nan_color: str = "#c7c7c7",
        **kwargs,
    ):
        if mode not in ("missing_missing", "present_present", "present_missing"):
            raise ValueError(
                "mode must be 'missing_missing', 'present_present', or 'present_missing'"
            )
        super().__init__(data=data, **kwargs)

        self.mode = mode
        self.selected_columns = selected_columns
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
    # Correlation matrix preparation
    # ------------------------------------------------------------------
    def _filter_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Apply column selection, constant-column dropping, ordering, and limit."""
        if self.selected_columns is not None:
            cols = [c for c in self.selected_columns if c in matrix.columns]
            if not cols:
                raise ValueError("No selected_columns found in DataFrame.")
            matrix = matrix[cols]

        if self.drop_constant_columns:
            constant = [c for c in matrix.columns if matrix[c].nunique() <= 1]
            if constant:
                matrix = matrix.drop(columns=constant)

        if self.order_by_missingness and not matrix.empty:
            ascending = self.order == "asc"
            miss_rate = self.data.col_missing_rate
            ordered = (
                miss_rate.loc[miss_rate.index.isin(matrix.columns)]
                .sort_values(ascending=ascending)
                .index
            )
            matrix = matrix[[c for c in ordered if c in matrix.columns]]

        if self.max_columns > 0 and matrix.shape[1] > self.max_columns:
            matrix = matrix.iloc[:, : self.max_columns]

        return matrix

    def _get_correlation_matrix(self) -> pd.DataFrame:
        if self.mode == "missing_missing":
            matrix = self._filter_matrix(self.data.mask_missing)
            if matrix.shape[1] < 2:
                raise ValueError(
                    "Not enough columns with varying missingness to compute correlation."
                )
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = matrix.corr()
            return corr.loc[corr.columns, corr.columns]

        elif self.mode == "present_present":
            matrix = self._filter_matrix(self.data.mask_present.astype(float))
            if matrix.shape[1] < 2:
                raise ValueError(
                    "Not enough columns with varying presence to compute correlation."
                )
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = matrix.corr()
            return corr.loc[corr.columns, corr.columns]

        else:  # present_missing
            corr = self.data.present_missing_corr

            if self.selected_columns is not None:
                cols = [c for c in self.selected_columns if c in corr.columns]
                if not cols:
                    raise ValueError("No selected_columns found in DataFrame.")
                corr = corr.loc[cols, cols]

            if self.drop_constant_columns:
                constant = [
                    c for c in self.data.mask_missing.columns
                    if self.data.mask_missing[c].nunique() <= 1
                ]
                corr = corr.drop(index=constant, columns=constant, errors="ignore")

            if self.order_by_missingness and not corr.empty:
                ascending = self.order == "asc"
                ordered = (
                    self.data.mask_missing.mean()
                    .sort_values(ascending=ascending)
                )
                ordered_cols = [c for c in ordered.index if c in corr.columns]
                corr = corr.loc[ordered_cols, ordered_cols]

            if self.max_columns > 0 and corr.shape[0] > self.max_columns:
                corr = corr.iloc[: self.max_columns, : self.max_columns]

            return corr

    # ------------------------------------------------------------------
    # Mode-specific labels
    # ------------------------------------------------------------------
    def _colorbar_config(self) -> dict:
        configs = {
            "missing_missing": dict(
                title=(
                    "Missingness correlation"
                    "<br><span style='font-size:10px'>NaN = insufficient overlap</span>"
                ),
                tickmode="array",
                tickvals=[-1, 0, 1],
                ticktext=["Mutually exclusive", "Independent", "Missing together"],
            ),
            "present_present": dict(
                title=(
                    "Presence correlation"
                    "<br><span style='font-size:10px'>NaN = constant column</span>"
                ),
                tickmode="array",
                tickvals=[-1, 0, 1],
                ticktext=["Mutually exclusive", "Independent", "Present together"],
            ),
            "present_missing": dict(
                title=(
                    "Present vs Missing"
                    "<br><span style='font-size:10px'>NaN = constant column</span>"
                ),
                tickmode="array",
                tickvals=[-1, 0, 1],
                ticktext=["Present together", "Independent", "Present → Missing"],
            ),
        }
        return configs[self.mode]

    def _hover_template(self) -> str:
        templates = {
            "missing_missing": (
                "<b>%{y}</b> vs <b>%{x}</b><br>"
                "Missingness correlation: %{z:.2f}<extra></extra>"
            ),
            "present_present": (
                "<b>%{y}</b> vs <b>%{x}</b><br>"
                "Presence correlation: %{z:.2f}<extra></extra>"
            ),
            "present_missing": (
                "<b>Present</b>: %{y}<br>"
                "<b>Missing</b>: %{x}<br>"
                "Correlation: %{z:.2f}<extra></extra>"
            ),
        }
        return templates[self.mode]

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        corr = self._get_correlation_matrix()

        effective_show_values = self.show_values and corr.shape[0] <= 30

        # Hide upper triangle when requested
        if not self.show_upper_triangle:
            tri_mask = pd.DataFrame(
                np.triu(np.ones(corr.shape, dtype=bool), k=1),
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
                val = rounded[idx]
                if np.isnan(val) or np.isclose(val, 0.0):
                    text[idx] = ""
                else:
                    text[idx] = val
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

        fig.add_trace(
            go.Heatmap(
                z=corr_values,
                x=x_labels,
                y=y_labels,
                colorscale=self.colorscale,
                zmin=-1,
                zmax=1,
                zmid=0,
                text=text if effective_show_values else None,
                texttemplate="%{text}" if effective_show_values else None,
                colorbar=self._colorbar_config() if self.show_colorbar else None,
                hovertemplate=self._hover_template(),
                hoverongaps=False,
            )
        )

        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(tickangle=0)
        self._apply_base_layout(fig)
        return fig
