import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, List, Literal

from missingfcup.plots.Plot import Plot
from missingfcup.core.MissingData import MissingData

class CorrelationHeatmap(Plot):
    """
    Heatmap showing correlation between column-level missingness.

    Each cell represents the correlation between two columns'
    missingness patterns.
    """

    def __init__(
        self,
        data: MissingData,
        selected_columns: Optional[List[str]] = None,
        colorscale: str = "RdBu",
        show_values: bool = True,
        max_columns: int = 30,
        drop_constant_columns: bool = True,
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

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        df = self.data.data
        missing_matrix = df.isna()

        if self.selected_columns is not None:
            cols = [c for c in self.selected_columns if c in missing_matrix.columns]
            if not cols:
                raise ValueError("No selected_columns found in DataFrame.")
            missing_matrix = missing_matrix[cols]

        # Drop constant-missingness columns
        if self.drop_constant_columns:
            constant_cols = [
                c for c in missing_matrix.columns
                if missing_matrix[c].nunique() <= 1
            ]
            if constant_cols:
                missing_matrix = missing_matrix.drop(columns=constant_cols)

        # Order by missingness fraction
        if self.order_by_missingness and not missing_matrix.empty:
            ascending = self.order == "asc"
            missing_fraction = missing_matrix.mean().sort_values(ascending=ascending)
            missing_matrix = missing_matrix[missing_fraction.index]

        # Column limit
        if self.max_columns > 0 and missing_matrix.shape[1] > self.max_columns:
            missing_matrix = missing_matrix.iloc[:, : self.max_columns]

        if missing_matrix.shape[1] < 2:
            raise ValueError(
                "Not enough columns with varying missingness to compute correlation."
            )

        corr = missing_matrix.corr()
        # Ensure columns/rows align (standard correlation matrix layout)
        corr = corr.loc[corr.columns, corr.columns]

        # Avoid unreadable text for large matrices
        effective_show_values = self.show_values and corr.shape[0] <= 30

        if not self.show_upper_triangle:
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
        # NaN overlay (gray)
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
                        "Missingness correlation"
                        "<br><span style='font-size:10px'>NaN = insufficient overlap</span>"
                    ),
                    tickmode="array",
                    tickvals=[-1, 0, 1],
                    ticktext=[
                        "Mutually exclusive missingness",
                        "Independent missingness",
                        "Missing together",
                    ],
                ) if self.show_colorbar else None,
                hovertemplate=(
                    "<b>%{y}</b> vs <b>%{x}</b><br>"
                    "Correlation: %{z:.1f}<extra></extra>"
                ),
                hoverongaps=False,
            )
        )

        # Axis labels intentionally empty (matrix-style plot)
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
        )

        # Apply shared layout/theme
        self._apply_base_layout(fig)

        return fig

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def fig(self) -> go.Figure:
        if self._figure is None:
            self._figure = self._build_figure()
        return self._figure

    def show(self):
        self.fig.show()

    def save(self, path: str):
        self.fig.write_html(path)
