import plotly.graph_objects as go
import pandas as pd
from typing import Optional

from .Plot import Plot
from ..core.MissingData import MissingData
from ..core.ViewMetadata import ViewMetadata

class PatternBarChart(Plot):
    """
    Bar chart showing frequency of row-level missingness patterns.

    Each bar represents a unique combination of missing variables.
    Height corresponds to number of rows with that pattern.
    """

    def __init__(
        self,
        data: MissingData,
        metadata: Optional[ViewMetadata] = None,
        figure_size_pixels: tuple[int, int] = (900, 500),
        bar_color: str = "#7f7f7f",
        title: Optional[str] = None,
        max_patterns: Optional[int] = None,
    ):
        self.data = data
        self.metadata = metadata

        self.width, self.height = figure_size_pixels
        self.bar_color = bar_color
        self.title = title
        self.max_patterns = max_patterns

        self._figure: Optional[go.Figure] = None

    # ------------------------------------------------------------------
    # Pattern computation
    # ------------------------------------------------------------------
    def _compute_patterns(self) -> pd.Series:
        df = self.data.data

        def row_pattern(row):
            missing = row.index[row.isna()].tolist()
            if not missing:
                return "No values missing"
            return ", ".join(missing)

        patterns = df.apply(row_pattern, axis=1)
        counts = patterns.value_counts()

        if self.max_patterns is not None:
            counts = counts.iloc[: self.max_patterns]

        return counts

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        counts = self._compute_patterns()

        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts.index.tolist(),
                    y=counts.values.tolist(),
                    marker_color=self.bar_color,
                    text=counts.values.tolist(),
                    textposition="outside",
                )
            ]
        )

        fig.update_layout(
            title=self.title or "Number of rows with same missing patterns",
            width=self.width,
            height=self.height,
            xaxis_title="Missingness pattern",
            yaxis_title="Number of rows",
            xaxis_tickangle=-45,
            margin=dict(t=60, b=120),
        )

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
