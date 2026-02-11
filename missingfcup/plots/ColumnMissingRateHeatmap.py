import plotly.graph_objects as go
import pandas as pd
from typing import Optional, List, Literal
import warnings

from .Plot import Plot
from ..core.MissingData import MissingData

class ColumnMissingRateHeatmap(Plot):
    """
    Heatmap showing missing rate per column.

    Single-row heatmap where each cell represents
    the fraction or percentage of missing values
    in a column.
    """

    def __init__(
        self,
        data: MissingData,
        selected_columns: Optional[List[str]] = None,
        scale: Literal["fraction", "percentage"] = "fraction",
        colorscale: str = "Reds",
        show_values: bool = True,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.scale = scale
        self.colorscale = colorscale
        self.show_values = show_values

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        rates = self.data.missingness_per_column

        if self.selected_columns is not None:
            missing = [c for c in self.selected_columns if c not in rates.index]
            if missing:
                warnings.warn(f"Skipping missing columns: {missing}")
            rates = rates.loc[
                [c for c in self.selected_columns if c in rates.index]
            ]

        if rates.empty:
            raise ValueError("No columns available to plot")

        if self.scale == "percentage":
            values = rates * 100
            label = "Missing (%)"
            text = [[f"{v:.2f}%" for v in values]]
        else:
            values = rates
            label = "Missing rate"
            text = [[f"{v:.2f}" for v in values]]

        zmin = 0
        zmax = max(values.max(), 1e-6)

        fig = go.Figure(
            data=go.Heatmap(
                z=[values.values],
                x=values.index.tolist(),
                y=["Missing rate"],
                colorscale=self.colorscale,
                zmin=zmin,
                zmax=zmax,
                text=text if self.show_values else None,
                texttemplate="%{text}" if self.show_values else None,
                showscale=True,
                colorbar=dict(title=label),
            )
        )

        fig.update_layout(
            xaxis_title="Columns",
            yaxis=dict(showticklabels=False),
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
