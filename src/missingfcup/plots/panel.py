import copy
import os
from typing import List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from missingfcup.plots._plot import _Plot, _slugify, _write_figure


class Panel:
    """
    Create a multi-panel display combining multiple plots in a grid layout.

    The Panel class automatically arranges plots in a grid (maximum `max_cols` columns)
    and handles sizing to prevent overlap. Colorbars and legends are hidden
    in the panel view to avoid clutter.

    Parameters
    ----------
    plots : Optional[List[_Plot]]
        Optional list of plot objects to combine (can also be added via ``add()``)
    title : Optional[str]
        Optional global title for the panel
    description : Optional[str]
        Optional subtitle/description shown below the title
    background_color : Optional[str]
        Panel background color
    text_color : Optional[str]
        Global text color
    max_cols : int
        Maximum number of columns (default: 3)
    max_plots : int
        Maximum number of plots allowed in a panel (default: 6)
    """

    def __init__(
        self,
        plots: Optional[List[_Plot]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        max_cols: int = 3,
        max_plots: int = 6,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self.plots: List[_Plot] = list(plots) if plots else []
        self.title = title or "Combined Plots"
        self.description = description
        self.background_color = background_color
        self.text_color = text_color
        self.max_cols = max_cols
        self.max_plots = max_plots
        self.width = width
        self.height = height

    def add(self, plot: _Plot) -> "Panel":
        """Add a plot to the panel. Returns self for chaining."""
        if len(self.plots) >= self.max_plots:
            raise ValueError(f"Panel already has {self.max_plots} plots (max_plots limit)")
        self.plots.append(plot)
        return self

    def _create_combined_figure(self) -> go.Figure:
        if not self.plots:
            raise ValueError(
                "Panel has no plots. Use add() or pass plots= to add plots before showing."
            )
        if len(self.plots) > self.max_plots:
            raise ValueError(f"Panel supports at most {self.max_plots} plots")

        n_plots = len(self.plots)

        cols = min(self.max_cols, n_plots)
        rows = (n_plots + cols - 1) // cols

        subplot_titles = []
        for i in range(rows * cols):
            if i < n_plots:
                subplot_titles.append(self.plots[i].title or "")
            else:
                subplot_titles.append("")

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles if any(subplot_titles) else None,
            horizontal_spacing=0.12,
            vertical_spacing=0.22,
        )

        for idx, plot in enumerate(self.plots):
            row = idx // cols + 1
            col = idx % cols + 1

            plot_fig = plot.fig

            for trace in plot_fig.data:
                # Deep copy so the original plot's traces are not mutated
                trace_copy = copy.deepcopy(trace)

                trace_copy.showlegend = False

                # Hide colorbars to prevent overlap
                if hasattr(trace_copy, "showscale") and trace_copy.showscale:
                    trace_copy.update(showscale=False)

                fig.add_trace(trace_copy, row=row, col=col)

            # Carry the axis settings across, not only the titles. Plots that draw on
            # numeric positions and name them with ticktext -- the matrix does this on
            # both axes -- would otherwise show bare indices once panelled.
            for source, update in (
                (plot_fig.layout.xaxis, fig.update_xaxes),
                (plot_fig.layout.yaxis, fig.update_yaxes),
            ):
                settings = {}
                if source.title.text:
                    settings["title_text"] = source.title.text
                if source.ticktext is not None:
                    settings["tickmode"] = "array"
                    settings["tickvals"] = source.tickvals
                    settings["ticktext"] = source.ticktext
                if source.tickangle is not None:
                    settings["tickangle"] = source.tickangle
                if settings:
                    update(row=row, col=col, **settings)

        if self.width is not None:
            total_width = self.width
        elif cols == 3:
            total_width = 1500
        elif cols == 2:
            total_width = 1400
        else:
            total_width = 800
        total_height = self.height if self.height is not None else 450 * rows + 150

        title_text = self.title
        if self.description:
            title_text = f"{self.title}<br><sub>{self.description}</sub>"

        fig.update_layout(
            title_text=title_text,
            width=total_width,
            height=total_height,
            showlegend=False,
            margin=dict(l=60, r=120, t=120, b=80),
            plot_bgcolor=self.background_color,
            paper_bgcolor=self.background_color,
            font=dict(color=self.text_color) if self.text_color else None,
        )

        return fig

    @property
    def _download_filename(self) -> str:
        if self.title and self.title != "Combined Plots":
            return "panel-" + _slugify(self.title)
        return "panel"

    def show(self):
        """Display all plots in a single interactive figure."""
        config = {"toImageButtonOptions": {"filename": self._download_filename}}
        self._create_combined_figure().show(config=config)

    def save(self, path: Optional[str] = None, save_individual: bool = False):
        """Save the panel. path is the destination file including extension (.html or .png).
        Defaults to plots/<name>.png relative to the current directory.
        If save_individual=True, each sub-plot is also saved as a separate PNG in the same directory."""
        fig = self._create_combined_figure()
        written = _write_figure(fig, path, self._download_filename)
        if save_individual:
            out_dir = os.path.dirname(written) or "plots"
            os.makedirs(out_dir, exist_ok=True)
            for plot in self.plots:
                individual_path = os.path.join(out_dir, f"{plot._download_filename}.png")
                plot.save(individual_path)
