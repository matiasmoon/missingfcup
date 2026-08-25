import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
from missingfcup.plots._plot import _Plot


class _Totals(_Plot):
    """
    Simple bar chart showing the total number of present and missing cells
    across the entire dataset.

    Useful when you just want the overall number.
    """

    def __init__(self, data: MissingData, *, show_values: bool = True, **kwargs):
        super().__init__(data=data, **kwargs)
        self.show_values = show_values

    def _build_figure(self) -> go.Figure:
        total_cells = self.data.data.size
        total_missing = self.data.total_missing_count
        total_present = total_cells - total_missing

        missing_pct = total_missing / total_cells * 100
        present_pct = total_present / total_cells * 100

        fig = go.Figure()

        fig.add_bar(
            x=["!NA", "NA"],
            y=[total_present, total_missing],
            marker_color=[self.present_color, self.missing_color],
            text=[
                f"{total_present:,}<br>({present_pct:.1f}%)",
                f"{total_missing:,}<br>({missing_pct:.1f}%)",
            ]
            if self.show_values
            else None,
            textposition="outside" if self.show_values else None,
            # Cells, not rows: this is the one plot counting the whole grid.
            hovertemplate=_hover.build(
                _hover.title("%{x}"),
                "%{y:,} of %{customdata[0]:,} cells (%{customdata[1]:.2f}%)",
            ),
            customdata=[[total_cells, present_pct], [total_cells, missing_pct]],
            # One trace holds both bars, so a legend entry would read 'trace 0'.
            # The x-axis already names them.
            showlegend=False,
        )

        max_y = max(total_present, total_missing)
        fig.update_layout(
            yaxis=dict(range=[0, max_y * 1.18]),
            showlegend=False,
        )
        fig.update_yaxes(title_text="Cells")

        self._apply_base_layout(fig)
        return fig
