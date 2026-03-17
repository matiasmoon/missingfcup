import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, List, Literal

from missingfcup.plots.Plot import Plot
from missingfcup.core.MissingData import MissingData


class AllCorrelationHeatmap(Plot):
    """
    Heatmap showing correlation between present indicators and
    missingness indicators of all columns.
    """

    def __init__(
        self,
        data: MissingData,
        selected_columns: Optional[List[str]] = None,
        colorscale: str = "RdBu",
        show_values: bool = True,
        max_columns: int = 30,
        drop_constant_columns: bool = False,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 1,
        show_colorbar: bool = True,
        show_upper_triangle: bool = True,
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

    def _build_figure(self) -> go.Figure:
        corr = self.data.present_missing_correlation

        if self.selected_columns is not None:
            cols = [c for c in self.selected_columns if c in corr.columns]
            if not cols:
                raise ValueError("No selected_columns found in DataFrame.")
            corr = corr.loc[cols, cols]

        if self.drop_constant_columns:
            missing_matrix = self.data.missing_mask
            constant_cols = [
                c for c in missing_matrix.columns
                if missing_matrix[c].nunique() <= 1
            ]
            if constant_cols:
                corr = corr.drop(index=constant_cols, columns=constant_cols, errors="ignore")

        if self.order_by_missingness and not corr.empty:
            ascending = self.order == "asc"
            missing_fraction = (
                self.data.missing_mask.mean().sort_values(ascending=ascending)
            )
            ordered = [c for c in missing_fraction.index if c in corr.columns]
            corr = corr.loc[ordered, ordered]

        if self.max_columns > 0 and corr.shape[0] > self.max_columns:
            corr = corr.iloc[: self.max_columns, : self.max_columns]

        if corr.shape[0] < 1 or corr.shape[1] < 1:
            raise ValueError("Not enough columns to compute correlation.")

        effective_show_values = self.show_values and corr.shape[0] <= 30

        if (
            not self.show_upper_triangle
            and corr.shape[0] == corr.shape[1]
            and list(corr.index) == list(corr.columns)
        ):
            mask = pd.DataFrame(
                np.triu(np.ones(corr.shape, dtype=bool), k=1),
                index=corr.index,
                columns=corr.columns,
            )
            corr = corr.mask(mask)

        x_labels = corr.columns.tolist()
        y_labels = corr.index.tolist()

        nan_mask = corr.isna().to_numpy()
        rounded = corr.round(self.value_round)
        if effective_show_values:
            text = rounded.astype(object).to_numpy()
            text[np.isclose(text, 0)] = ""
        else:
            text = None

        fig = go.Figure()
        if nan_mask.any():
            nan_layer = np.where(nan_mask, 1.0, np.nan)
            fig.add_trace(
                go.Heatmap(
                    z=nan_layer,
                    x=x_labels,
                    y=y_labels,
                    colorscale=[[0, self.nan_color], [1, self.nan_color]],
                    showscale=False,
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=x_labels,
                y=y_labels,
                colorscale=self.colorscale,
                zmin=-1,
                zmax=1,
                zmid=0,
                text=text if effective_show_values else None,
                texttemplate="%{text}" if effective_show_values else None,
                colorbar=dict(
                    title=(
                        "Present vs Missing correlation"
                        "<br><span style='font-size:10px'>NaN = constant column</span>"
                    ),
                    tickmode="array",
                    tickvals=[-1, 0, 1],
                    ticktext=[
                        "Present together",
                        "Independent",
                        "Present vs Missing",
                    ],
                ) if self.show_colorbar else None,
                hovertemplate=(
                    "<b>Present</b>: %{y}<br>"
                    "<b>Missing</b>: %{x}<br>"
                    "Correlation: %{z:.1f}<extra></extra>"
                ),
                hoverongaps=False,
            )
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
        )

        self._apply_base_layout(fig)
        return fig
