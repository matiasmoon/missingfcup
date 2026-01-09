from typing import List
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .plot import Plot

class Panel:
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
            # Disable individual legends to prevent overlap
            for trace in plot_fig.data:
                trace_copy = trace
                trace_copy.showlegend = False  # Hide legends in subplots
                fig.add_trace(trace_copy, row=row, col=col)

            # Copy axis labels if they exist
            if plot_fig.layout.xaxis.title.text:
                fig.update_xaxes(title_text=plot_fig.layout.xaxis.title.text,
                               row=row, col=col)
            if plot_fig.layout.yaxis.title.text:
                fig.update_yaxes(title_text=plot_fig.layout.yaxis.title.text,
                               row=row, col=col)

        # Update overall layout with better spacing
        fig.update_layout(
            height=500 * rows,  # Further increased height per row for better spacing
            showlegend=False,   # Disable legend to prevent overlap
            title_text="Combined Plots",
            margin=dict(l=60, r=60, t=100, b=80)  # Increased top and bottom margins
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