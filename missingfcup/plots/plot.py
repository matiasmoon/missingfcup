from abc import ABC, abstractmethod
from typing import Optional
import plotly.graph_objects as go
from ..core.MissingData import MissingData


class Plot(ABC):
    """Abstract base class for all visualizations."""

    def __init__(
        self,
        data: MissingData,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 500,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        missing_color: str = "#d62728",
        present_color: str = "#2ca02c",
        show_legend: bool = True,
        legend_title: Optional[str] = None,
    ):
        self.data = data
        self.title = title

        # Layout / theme
        self.width = width
        self.height = height
        self.background_color = background_color
        self.text_color = text_color

        # Semantic colors
        self.missing_color = missing_color
        self.present_color = present_color

        # Legend
        self.show_legend = show_legend
        self.legend_title = legend_title

        self._figure: Optional[go.Figure] = None

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_figure(self) -> go.Figure:
        """Subclasses must construct and return a plotly Figure."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _apply_base_layout(self, fig: go.Figure):
        """Apply shared layout, colors, and typography."""
        fig.update_layout(
            title=self.title,
            width=self.width,
            height=self.height,
            showlegend=self.show_legend,
            legend_title=self.legend_title,
            plot_bgcolor=self.background_color,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color) if self.text_color else None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def fig(self) -> go.Figure:
        """Lazily build and cache the figure."""
        if self._figure is None:
            self._figure = self._build_figure()
        return self._figure

    def show(self):
        """Display the figure."""
        self.fig.show()

    def save(self, path: str):
        """Save the figure as an HTML file."""
        self.fig.write_html(path)
