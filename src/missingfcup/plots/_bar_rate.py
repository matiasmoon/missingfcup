from typing import Literal

import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
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
        scale: Literal["fraction", "percentage"] = "fraction",
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)
        self.scale = scale

    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()
        columns = df.columns.tolist()

        rates = self.data.col_missing_rate.loc[columns]

        if self.scale == "percentage":
            values = rates * 100
            y_title = "Missing (%)"
            text_vals = [f"{v:.1f}%" for v in values]
        else:
            values = rates
            y_title = "Missing rate"
            text_vals = [f"{v:.2f}" for v in values]

        fig = go.Figure()

        fig.add_bar(
            x=columns if self.orientation == "vertical" else values,
            y=values if self.orientation == "vertical" else columns,
            name="Missing",
            marker_color=self.missing_color,
            text=text_vals if self.show_values else None,
            textposition="auto" if self.show_values else None,
        )

        first_col = columns[0] if columns else ""
        if self.orientation == "vertical":
            fig.update_xaxes(tickangle=-45, title_text=first_col)
            fig.update_yaxes(title_text=y_title)
        else:
            fig.update_xaxes(title_text=y_title)
            fig.update_yaxes(title_text=first_col)
        self._apply_base_layout(fig)
        return fig
