from typing import List, Optional
import copy
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from missingfcup.plots._plot import _Plot

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

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _create_combined_figure(self) -> go.Figure:
        if not self.plots:
            raise ValueError("Panel has no plots. Use add() or pass plots= to add plots before showing.")
        if len(self.plots) > self.max_plots:
            raise ValueError(f"Panel supports at most {self.max_plots} plots")

        n_plots = len(self.plots)

        # Grid layout
        cols = min(self.max_cols, n_plots)
        rows = (n_plots + cols - 1) // cols

        # Collect per-subplot titles (pad to full grid size)
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
                # Deep copy to avoid mutating the original plot's traces
                trace_copy = copy.deepcopy(trace)

                # No shared global legend in panels
                trace_copy.showlegend = False

                # Hide colorbars to prevent overlap
                if hasattr(trace_copy, "showscale") and trace_copy.showscale:
                    trace_copy.update(showscale=False)

                fig.add_trace(trace_copy, row=row, col=col)

            # Preserve axis titles
            if plot_fig.layout.xaxis.title.text:
                fig.update_xaxes(
                    title_text=plot_fig.layout.xaxis.title.text,
                    row=row,
                    col=col,
                )

            if plot_fig.layout.yaxis.title.text:
                fig.update_yaxes(
                    title_text=plot_fig.layout.yaxis.title.text,
                    row=row,
                    col=col,
                )

        # Figure sizing — use explicit override if provided, else fall back to heuristic
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def show(self):
        """Display all plots in a single interactive figure."""
        self._create_combined_figure().show()

    def save(self, path: str):
        """Save combined plots as a single HTML file."""
        self._create_combined_figure().write_html(path)
