import plotly.graph_objects as go
import pandas as pd
from typing import Optional, List
import warnings

from .Plot import Plot
from ..core.MissingData import MissingData

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
        max_columns: Optional[int] = 50,
        drop_constant_columns: bool = True,
        order_by_missingness: bool = True,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.colorscale = colorscale
        self.show_values = show_values
        self.max_columns = max_columns
        self.drop_constant_columns = drop_constant_columns
        self.order_by_missingness = order_by_missingness

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        df = self.data.data
        missing_matrix = df.isna()

        if self.selected_columns is not None:
            missing_matrix = missing_matrix[self.selected_columns]

        # Drop constant-missingness columns
        if self.drop_constant_columns:
            constant_cols = [
                c for c in missing_matrix.columns
                if missing_matrix[c].nunique() <= 1
            ]
            if constant_cols:
                missing_matrix = missing_matrix.drop(columns=constant_cols)
                warnings.warn(
                    f"Dropping columns with constant missingness: {constant_cols}",
                    UserWarning,
                )

        # Order by missingness fraction
        if self.order_by_missingness and not missing_matrix.empty:
            missing_fraction = missing_matrix.mean().sort_values(ascending=False)
            missing_matrix = missing_matrix[missing_fraction.index]

        # Column limit
        if self.max_columns is not None and missing_matrix.shape[1] > self.max_columns:
            missing_matrix = missing_matrix.iloc[:, : self.max_columns]

        corr = missing_matrix.corr()

        # Avoid unreadable text for large matrices
        effective_show_values = self.show_values and corr.shape[0] <= 30

        fig = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale=self.colorscale,
                zmin=-1,
                zmax=1,
                zmid=0,
                text=corr.round(2).values if effective_show_values else None,
                texttemplate="%{text}" if effective_show_values else None,
                colorbar=dict(title="Missingness correlation"),
                hovertemplate=(
                    "<b>%{y}</b> vs <b>%{x}</b><br>"
                    "Correlation: %{z:.2f}<extra></extra>"
                ),
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
