from typing import List
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .Plot import Plot


class Panel:
    """
    Create a multi-panel display combining multiple plots in a grid layout.

    The Panel class automatically arranges plots in a grid (maximum 2 columns)
    and handles sizing to prevent overlap. Colorbars are automatically hidden
    in the panel view to prevent overlap - use descriptive plot titles to
    indicate what each visualization shows.

    Parameters
    ----------
    plots : List[Plot]
        List of plot objects (e.g., Heatmap instances) to combine

    Examples
    --------
    >>> panel = Panel([
    ...     missing.heatmap(group_by_mode="binary", title="Binary Mode"),
    ...     missing.heatmap(group_by_mode="completeness", title="Completeness Mode")
    ... ])
    >>> panel.show()
    """

    def __init__(self, plots: List[Plot]):
        if not plots:
            raise ValueError("Panel requires at least one plot")
        self.plots = plots

    def _create_combined_figure(self) -> go.Figure:
        """Create a single figure with all plots arranged as subplots"""
        n_plots = len(self.plots)

        # Grid layout: max 2 columns
        cols = min(2, n_plots)
        rows = (n_plots + cols - 1) // cols

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                getattr(plot, "title", f"Plot {i + 1}")
                for i, plot in enumerate(self.plots)
            ],
            horizontal_spacing=0.15,
            vertical_spacing=0.25,
        )

        for idx, plot in enumerate(self.plots):
            row = idx // cols + 1
            col = idx % cols + 1

            # Retrieve plotly figure
            plot_fig = plot.fig if hasattr(plot, "fig") else plot.get_figure()

            for trace in plot_fig.data:
                # Copy trace reference safely
                trace_copy = trace

                # Hide legends inside panels
                trace_copy.showlegend = False

                # Hide colorbars to prevent overlap
                if hasattr(trace_copy, "showscale") and trace_copy.showscale:
                    trace_copy.update(showscale=False)

                fig.add_trace(trace_copy, row=row, col=col)

            # Preserve axis titles if present
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

        # Figure sizing logic
        if cols == 2:
            total_width = 1400
        else:
            total_width = 800

        total_height = 450 * rows + 150

        fig.update_layout(
            width=total_width,
            height=total_height,
            showlegend=False,
            title_text="Combined Plots",
            margin=dict(l=60, r=120, t=100, b=80),
        )

        return fig

    def show(self):
        """Display all plots in a single interactive figure"""
        self._create_combined_figure().show()

    def save(self, path: str):
        """Save combined plots as a single HTML file"""
        self._create_combined_figure().write_html(path)