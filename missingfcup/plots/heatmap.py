import plotly.graph_objects as go
from typing import Optional

from .plot import Plot
from ..core.matrix import Matrix

class Heatmap(Plot):
    """Interactive missingness heatmap."""

    def __init__(
        self,
        matrix: Matrix,
        height: int = 600,
        width: int = 900,
        present_color: str = "#2ca02c",
        missing_color: str = "#d62728",
        hover_template: Optional[str] = None,
        show_hover: bool = True,
        show_scale: bool = False,
        column_gap: float = 0.5,
        title: Optional[str] = None,
    ):
        self.matrix = matrix
        self.height = height
        self.width = width
        self.present_color = present_color
        self.missing_color = missing_color
        self.hover_template = hover_template
        self.show_hover = show_hover
        self.show_scale = show_scale
        self.column_gap = column_gap
        self.title = title

        self._figure: Optional[go.Figure] = None

    def _build_figure(self) -> go.Figure:
        hover_template = self.hover_template or (
            "<b>Row</b>: %{y}<br>"
            "<b>Col</b>: %{x}<br>"
            "<b>Present?</b>: %{z}<extra></extra>"
        )

        fig = go.Figure(
            go.Heatmap(
                z=self.matrix.values,
                x=self.matrix.columns,
                y=list(range(len(self.matrix.values))),
                colorscale=[[0, self.missing_color], [1, self.present_color]],
                showscale=self.show_scale,
                hoverinfo="text" if self.show_hover else "skip",
                hovertemplate=hover_template,
                xgap=self.column_gap,
            )
        )

        fig.update_layout(
            title=self.title or "Interactive Missingness Matrix",
            xaxis_title="Columns",
            yaxis_title="Row Index",
            height=self.height,
            width=self.width,
        )

        return fig

    @property
    def fig(self) -> go.Figure:
        """Get or build the figure."""
        if self._figure is None:
            self._figure = self._build_figure()
        return self._figure

    def show(self):
        if self._figure is None:
            self._figure = self._build_figure()
        self._figure.show()

    def save(self, path: str):
        if self._figure is None:
            self._figure = self._build_figure()
        self._figure.write_html(path)
