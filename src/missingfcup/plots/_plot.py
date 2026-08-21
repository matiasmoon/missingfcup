import os
import re
from abc import ABC, abstractmethod
from typing import Optional

import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData


def _write_figure(fig: go.Figure, path: Optional[str], default_name: str) -> str:
    """Write ``fig`` to ``path``; the extension picks the format (``.png`` or ``.html``).

    ``path=None`` defaults to ``plots/<default_name>.png``. Returns the path written,
    so callers (Plot and Panel) share one place for this file-writing logic.
    """
    if path is None:
        path = os.path.join("plots", f"{default_name}.png")
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "html"
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    if ext == "png":
        fig.write_image(path)
    else:
        fig.write_html(path)
    return path


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
        max_label_length: int = 48,
    ):
        self.data = data
        self.title = title

        self.width = min(width, 2000)
        self.height = min(height, 1000)
        self.background_color = background_color
        self.text_color = text_color

        self.missing_color = missing_color
        self.present_color = present_color

        self.show_legend = show_legend
        self.max_label_length = max_label_length

        self._figure: Optional[go.Figure] = None

    @abstractmethod
    def _build_figure(self) -> go.Figure:
        """Subclasses must construct and return a plotly Figure."""
        raise NotImplementedError

    def _apply_base_layout(self, fig: go.Figure):
        """Apply shared layout, colors, and typography."""
        fig.update_layout(
            title=self.title,
            width=self.width,
            height=self.height,
            showlegend=self.show_legend,
            plot_bgcolor=self.background_color,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color) if self.text_color else None,
        )

    def _truncate_labels(
        self,
        labels: list,
        *,
        min_len: int = 16,
        width_divisor: int = 12,
        ellipsis: str = "…",
    ) -> list:
        """Shorten long axis labels to fit, then disambiguate any duplicates the
        shortening created. Shared by the plots that put column names on an axis.

        ``max_label_length`` overrides the width-based budget when set. ``min_len``,
        ``width_divisor`` and ``ellipsis`` let a caller match its own layout (the
        UpSet plot has a narrower label column, for example).
        """
        if self.max_label_length > 0:
            max_len = self.max_label_length
        else:
            max_len = max(min_len, int(self.width / width_divisor))

        def truncate(label: str) -> str:
            if max_len <= 0 or len(label) <= max_len:
                return label
            return label[: max_len - len(ellipsis)] + ellipsis

        out = [truncate(label) for label in labels]

        if len(set(out)) < len(out):
            # Two columns whose names differ only past the cut come out identical, and
            # plotly treats identical category names as one category. The marker has
            # to be visible: padding with spaces would make them distinct to plotly
            # while still reading as duplicates on screen. It also has to come out of
            # the budget rather than be added on top of it, or the cap is not a cap.
            counts_seen: dict = {}
            adjusted = []
            for label in out:
                counts_seen[label] = counts_seen.get(label, 0) + 1
                idx = counts_seen[label]
                if idx == 1:
                    adjusted.append(label)
                    continue
                marker = f"~{idx}"
                keep = max(0, max_len - len(marker)) if max_len > 0 else len(label)
                adjusted.append(label[:keep] + marker)
            out = adjusted

        return out

    @property
    def _download_filename(self) -> str:
        parts = [_class_to_kebab(self.__class__.__name__)]
        if self.title:
            parts.append(_slugify(self.title))
        return "-".join(parts)

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

    def _ipython_display_(self):
        """Render inline in notebooks when the plot is the last expression.

        Lets the flat facade (e.g. ``mf.matrix(df)``) auto-render like missingno
        without a forced ``.show()`` side effect in scripts.
        """
        self.show()

    def save(self, path: Optional[str] = None):
        """Save the figure to ``path``, where the extension picks the format (.html or .png).
        Defaults to plots/<name>.png relative to the current directory."""
        _write_figure(self.fig, path, self._download_filename)
