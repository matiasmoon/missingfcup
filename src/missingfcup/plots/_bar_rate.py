from typing import Literal

import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
from missingfcup.plots._bar_base import _BarBase


class _BarRate(_BarBase):
    """
    Bar chart of missing rate per column.

    Shows the fraction or percentage of missing values for each column.
    """

    def __init__(
        self,
        data: MissingData,
        *,
        measure: Literal["fraction", "percentage"] = "fraction",
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)
        self.measure = measure

    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()
        columns = df.columns.tolist()

        rates = self.data.col_missing_rate.loc[columns]
        full_names = list(columns)
        columns = self._truncate_labels(columns)
        total = len(self.data.data)
        counts = self.data.col_missing_count.loc[full_names]

        if self.measure == "percentage":
            values = rates * 100
            y_title = "Missing (%)"
            text_vals = [f"{v:.2f}%" for v in values]
        else:
            values = rates
            y_title = "Missing rate"
            text_vals = [f"{v:.2f}" for v in values]

        fig = go.Figure()

        # Only ever one series, so a legend entry would say nothing.
        fig.add_bar(
            showlegend=False,
            x=columns if self.orientation == "vertical" else values,
            y=values if self.orientation == "vertical" else columns,
            name="NA",
            marker_color=self.missing_color,
            text=text_vals if self.show_values else None,
            textposition="auto" if self.show_values else None,
            # A rate alone hides how much data is behind it, so the count comes too.
            customdata=_hover.customdata(
                full_names,
                [_hover.rate(v, self.measure == "percentage") for v in values],
                [_hover.rows_of_total(c, total) for c in counts],
            ),
            hovertemplate=_hover.build(
                _hover.title("%{customdata[0]}"),
                "NA: %{customdata[1]}",
                "%{customdata[2]}",
            ),
        )

        if self.orientation == "vertical":
            fig.update_xaxes(tickangle=-45, title_text="Column")
            fig.update_yaxes(title_text=y_title)
        else:
            fig.update_xaxes(title_text=y_title)
            fig.update_yaxes(title_text="Column")
        self._apply_base_layout(fig)
        return fig
