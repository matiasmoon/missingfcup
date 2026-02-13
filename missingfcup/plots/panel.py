from typing import List, Optional
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from .Plot import Plot

class Panel:
    """
    Create a multi-panel display combining multiple plots in a grid layout.

    The Panel class automatically arranges plots in a grid (maximum 2 columns)
    and handles sizing to prevent overlap. Colorbars and legends are hidden
    in the panel view to avoid clutter.

    Parameters
    ----------
    plots : List[Plot]
        List of plot objects to combine
    title : Optional[str]
        Optional global title for the panel
    background_color : Optional[str]
        Panel background color
    text_color : Optional[str]
        Global text color
    max_cols : int
        Maximum number of columns (default: 2)
    max_plots : int
        Maximum number of plots allowed in a panel (default: 4)
    """

    def __init__(
        self,
        plots: List[Plot],
        title: Optional[str] = None,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        max_cols: int = 2,
        max_plots: int = 4,
    ):
        if not plots:
            raise ValueError("Panel requires at least one plot")
        if len(plots) > max_plots:
            raise ValueError(f"Panel supports at most {max_plots} plots")

        self.plots = plots
        self.title = title or "Combined Plots"
        self.background_color = background_color
        self.text_color = text_color
        self.max_cols = max_cols
        self.max_plots = max_plots

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _create_combined_figure(self) -> go.Figure:
        n_plots = len(self.plots)

        # Grid layout
        if n_plots <= 2:
            cols = n_plots
        elif n_plots == 3:
            cols = 3
        else:
            cols = min(self.max_cols, n_plots)
        rows = (n_plots + cols - 1) // cols

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                plot.title or f"Plot {i + 1}"
                for i, plot in enumerate(self.plots)
            ],
            horizontal_spacing=0.12,
            vertical_spacing=0.22,
        )

        for idx, plot in enumerate(self.plots):
            row = idx // cols + 1
            col = idx % cols + 1

            plot_fig = plot.fig

            for trace in plot_fig.data:
                # Shallow copy is fine; Plotly handles trace reuse safely
                trace_copy = trace

                # Hide legends inside panels
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

        # Figure sizing heuristic
        if cols == 3:
            total_width = 1500
        elif cols == 2:
            total_width = 1400
        else:
            total_width = 800
        total_height = 450 * rows + 150

        fig.update_layout(
            title_text=self.title,
            width=total_width,
            height=total_height,
            showlegend=False,
            margin=dict(l=60, r=120, t=100, b=80),
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
