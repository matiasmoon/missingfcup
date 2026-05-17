import plotly.graph_objects as go

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData


class _BarchartTotal(_Plot):
    """
    Simple bar chart showing the total number of present and missing cells
    across the entire dataset.

    Useful as a one-glance summary of the overall data completeness.
    """

    def __init__(
        self,
        data: MissingData,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

    def _build_figure(self) -> go.Figure:
        total_cells   = self.data.data.size
        total_missing = self.data.total_missing_count
        total_present = total_cells - total_missing

        missing_pct = total_missing / total_cells * 100
        present_pct = total_present / total_cells * 100

        fig = go.Figure()

        fig.add_bar(
            x=["Present", "Missing"],
            y=[total_present, total_missing],
            marker_color=[self.present_color, self.missing_color],
            text=[
                f"{total_present:,}<br>({present_pct:.1f}%)",
                f"{total_missing:,}<br>({missing_pct:.1f}%)",
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Count: %{y:,}<br>"
                "Percent: %{customdata:.2f}%<extra></extra>"
            ),
            customdata=[present_pct, missing_pct],
        )

        max_y = max(total_present, total_missing)
        fig.update_layout(
            yaxis=dict(range=[0, max_y * 1.18]),
            showlegend=False,
        )
        fig.update_yaxes(title_text="Count")

        self._apply_base_layout(fig)
        return fig
