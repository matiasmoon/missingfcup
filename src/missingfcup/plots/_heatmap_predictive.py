import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, List, Literal

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData


class _HeatmapPredictive(_Plot):
    """
    Heatmap of present-vs-missing correlation.

    Each cell shows the correlation between being observed in one column
    versus missing in another (the present_missing mode).
    """

    def __init__(
        self,
        data: MissingData,
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
        super().__init__(data=data, **kwargs)

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
    def _get_correlation_matrix(self) -> pd.DataFrame:
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
        return dict(
            title=(
                "Present vs Missing"
                "<br><span style='font-size:10px'>blue = present → missing | red = both present</span>"
                "<br><span style='font-size:10px'>NaN = constant column</span>"
            ),
            tickmode="array",
            tickvals=[-1, 0, 1],
            ticktext=["Both present", "Independent", "Present → missing"],
        )

    def _hover_template(self) -> str:
        return (
            "<b>Present</b>: %{y}<br>"
            "<b>Missing</b>: %{x}<br>"
            "Correlation: %{z:.2f}<extra></extra>"
        )

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        corr = self._get_correlation_matrix()

        effective_show_values = self.show_values and corr.shape[0] <= 30

        # Show only upper triangle when requested; mask the strict lower triangle
        if self.show_upper_triangle:
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
                showscale=self.show_colorbar,
                colorbar=self._colorbar_config() if self.show_colorbar else None,
                hovertemplate=self._hover_template(),
                hoverongaps=False,
            )
        )

        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(tickangle=0, title_standoff=15)
        self._apply_base_layout(fig)
        return fig
