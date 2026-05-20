import re
from abc import ABC, abstractmethod
from typing import Optional
import plotly.graph_objects as go
from missingfcup.core.missing_data import MissingData


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _class_to_kebab(cls_name: str) -> str:
    name = cls_name.lstrip("_")
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


class _Plot(ABC):
    """
    Abstract base class for all visualizations.
    """

    def __init__(
        self,
        data: MissingData,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        missing_color: str = "#d62728",
        present_color: str = "#2ca02c",
        show_legend: bool = True,
        legend_title: Optional[str] = None,
        max_label_length: int = 48,
    ):
        self.data = data
        self.title = title

        # Layout / theme
        self.width = min(width, 2000)
        self.height = min(height, 1000)
        self.background_color = background_color
        self.text_color = text_color

        # Semantic colors
        self.missing_color = missing_color
        self.present_color = present_color

        # Legend
        self.show_legend = show_legend
        self.legend_title = legend_title
        self.max_label_length = max_label_length

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

    @property
    def _download_filename(self) -> str:
        parts = [_class_to_kebab(self.__class__.__name__)]
        if self.title:
            parts.append(_slugify(self.title))
        return "-".join(parts)

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
        config = {"toImageButtonOptions": {"filename": self._download_filename}}
        self.fig.show(config=config)

    def save(self, path: str = None):
        """Save the figure. path is the destination file including extension (.html or .png).
        Defaults to plots/<name>.html relative to the current directory."""
        import os
        if path is None:
            path = os.path.join("plots", f"{self._download_filename}.png")
        ext = os.path.splitext(path)[1].lstrip(".").lower() or "html"
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        if ext == "png":
            self.fig.write_image(path)
        else:
            self.fig.write_html(path)
