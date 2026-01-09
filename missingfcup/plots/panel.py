from typing import List
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .plot import Plot

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
    ...     missing.heatmap(sort_by_columns="category", group_by_mode="binary", title="Binary Mode"),
    ...     missing.heatmap(sort_by_columns="category", group_by_mode="completeness", title="Completeness Mode")
    ... ])
    >>> panel.show()

    Notes
    -----
    - Panel automatically controls figure dimensions (overrides individual plot sizes)
    - Colorbars are hidden to prevent overlap in multi-plot layouts
    - Use clear, descriptive titles for each subplot
    - Maximum 2 columns, rows adjust automatically based on number of plots
    """
    def __init__(self, plots: List[Plot]):
        self.plots = plots

    def _create_combined_figure(self):
        """Create a single figure with all plots as subplots"""
        n_plots = len(self.plots)

        # Create subplots - adjust rows/cols as needed
        # This creates a grid layout
        cols = min(2, n_plots)  # Max 2 columns
        rows = (n_plots + cols - 1) // cols  # Calculate needed rows

        # Improved spacing to prevent overlap
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[plot.title if hasattr(plot, 'title') else f'Plot {i+1}'
                          for i, plot in enumerate(self.plots)],
            horizontal_spacing=0.15,  # Increased horizontal spacing between plots
            vertical_spacing=0.25     # Increased vertical spacing between rows to prevent overlap
        )

        # Add each plot's traces to the combined figure
        for idx, plot in enumerate(self.plots):
            row = idx // cols + 1
            col = idx % cols + 1

            # Get the figure from the plot
            plot_fig = plot.fig if hasattr(plot, 'fig') else plot.get_figure()

            # Add all traces from this plot
            # Keep individual colorbars but position them correctly
            for trace in plot_fig.data:
                trace_copy = trace
                trace_copy.showlegend = False  # Hide legends in subplots

                # If this trace has a colorbar, adjust its position for the subplot
                if hasattr(trace_copy, 'showscale') and trace_copy.showscale:
                    # Hide colorbars in subplots - they cause overlap issues
                    # Users can refer to the individual plot titles to understand the visualization
                    trace_copy.update(showscale=False)

                fig.add_trace(trace_copy, row=row, col=col)

            # Copy axis labels if they exist
            if plot_fig.layout.xaxis.title.text:
                fig.update_xaxes(title_text=plot_fig.layout.xaxis.title.text,
                               row=row, col=col)
            if plot_fig.layout.yaxis.title.text:
                fig.update_yaxes(title_text=plot_fig.layout.yaxis.title.text,
                               row=row, col=col)

        # Calculate appropriate figure dimensions
        # For 2-column layout, we need extra width for colorbars between and on the right
        # For single column, we need width for plot + colorbar on right
        if cols == 2:
            # Two columns: each plot gets ~45% width, colorbars get the rest
            total_width = 1400  # Enough for 2 plots + colorbars
        else:
            # Single column: plot gets most width, colorbar on right
            total_width = 800

        # Height scales with rows, with extra space for titles and margins
        total_height = 450 * rows + 150  # Base height per row + overhead

        # Update overall layout with better spacing
        fig.update_layout(
            height=total_height,
            width=total_width,
            showlegend=False,   # Disable legend to prevent overlap
            title_text="Combined Plots",
            margin=dict(l=60, r=120, t=100, b=80)  # Extra right margin for colorbars
        )

        return fig
    
    def show(self):
        """Display all plots in a single figure"""
        fig = self._create_combined_figure()
        fig.show()
    
    def save(self, path: str):
        """Save combined plots as a single HTML file"""
        fig = self._create_combined_figure()
        fig.write_html(path)