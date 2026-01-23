import plotly.graph_objects as go
import pandas as pd
from typing import Optional

from .Plot import Plot
from ..core.MissingData import MissingData
from ..core.ViewMetadata import ViewMetadata

class ScatterPlot(Plot):
    """
    Scatter plot of two columns, colored by row-level missingness.

    A point is considered "Missing" if either x or y is missing.
    """

    def __init__(
        self,
        data: MissingData,
        x: str,
        y: str,
        metadata: Optional[ViewMetadata] = None,
        figure_size_pixels: tuple[int, int] = (900, 600),
        missing_color: str = "#F8766D",
        present_color: str = "#00BFC4",
        point_size: int = 8,
        title: Optional[str] = None,
    ):
        self.data = data
        self.x = x
        self.y = y
        self.metadata = metadata

        self.width, self.height = figure_size_pixels
        self.missing_color = missing_color
        self.present_color = present_color
        self.point_size = point_size
        self.title = title

        self._figure: Optional[go.Figure] = None


    def _build_figure(self) -> go.Figure:
        df = self.data.data

        if self.metadata is not None:
            df = self.metadata.apply(df)

        if self.x not in df.columns or self.y not in df.columns:
            raise ValueError(f"Columns '{self.x}' and '{self.y}' must exist")

        df = df[[self.x, self.y]].copy()

        # Masks
        x_missing = df[self.x].isna()
        y_missing = df[self.y].isna()

        # Compute offsets (10% below data range)
        def offset(series: pd.Series) -> float:
            s = series.dropna()
            data_range = s.max() - s.min()
            return s.min() - 0.1 * data_range

        x_offset = offset(df[self.x])
        y_offset = offset(df[self.y])

        # Replace missings with offsets
        plot_x = df[self.x].copy()
        plot_y = df[self.y].copy()

        plot_x[x_missing] = x_offset
        plot_y[y_missing] = y_offset

        fig = go.Figure()

        # Fully observed points
        observed = ~(x_missing | y_missing)
        fig.add_scatter(
            x=plot_x[observed],
            y=plot_y[observed],
            mode="markers",
            name="Not Missing",
            marker=dict(
                color=self.present_color,
                size=self.point_size,
            ),
            hovertemplate=(
                f"<b>{self.x}</b>: %{{x}}<br>"
                f"<b>{self.y}</b>: %{{y}}<extra></extra>"
            ),
        )

        # Missing points (x, y, or both)
        missing = x_missing | y_missing
        fig.add_scatter(
            x=plot_x[missing],
            y=plot_y[missing],
            mode="markers",
            name="Missing",
            marker=dict(
                color=self.missing_color,
                size=self.point_size,
                symbol="circle-open",
            ),
            hovertemplate=(
                f"<b>{self.x}</b>: %{{x}}<br>"
                f"<b>{self.y}</b>: %{{y}}<br>"
                "<b>Status</b>: Missing<extra></extra>"
            ),
        )

        fig.update_layout(
            title=self.title or f"{self.y} vs {self.x} (missing shown as offsets)",
            xaxis_title=self.x,
            yaxis_title=self.y,
            width=self.width,
            height=self.height,
            legend_title="missing",
        )

        # Expand axes so offsets are visible
        fig.update_xaxes(range=[x_offset, df[self.x].max()])
        fig.update_yaxes(range=[y_offset, df[self.y].max()])

        return fig


    @property
    def fig(self) -> go.Figure:
        if self._figure is None:
            self._figure = self._build_figure()
        return self._figure
