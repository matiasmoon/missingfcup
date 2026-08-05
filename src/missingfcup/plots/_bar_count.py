from typing import Literal

import pandas as pd
import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots._bar_base import _BarBase


class _BarCount(_BarBase):
    """
    Bar chart of missing and/or present counts per column.
    """

    def __init__(
        self,
        data: MissingData,
        *,
        value: Literal["missing", "present"] = "missing",
        show_both: bool = False,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)
        self.value = value
        self.show_both = show_both

    def _compute_missing_values(self, df: pd.DataFrame) -> pd.Series:
        return self.data.col_missing_count.loc[df.columns]

    def _compute_present_values(self, df: pd.DataFrame) -> pd.Series:
        return self.data.col_present_count.loc[df.columns]

    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()
        missing_values = self._compute_missing_values(df)
        present_values = self._compute_present_values(df)
        columns = missing_values.index.tolist()

        fig = go.Figure()

        if self.show_both:
            fig.add_bar(
                x=columns if self.orientation == "vertical" else missing_values,
                y=missing_values if self.orientation == "vertical" else columns,
                name="Missing",
                marker_color=self.missing_color,
                text=[f"{int(v)}" if self.show_values else None for v in missing_values],
                textposition="auto" if self.show_values else None,
            )
            fig.add_bar(
                x=columns if self.orientation == "vertical" else present_values,
                y=present_values if self.orientation == "vertical" else columns,
                name="Present",
                marker_color=self.present_color,
                text=[f"{int(v)}" if self.show_values else None for v in present_values],
                textposition="auto" if self.show_values else None,
            )
            fig.update_layout(barmode="stack")
        else:
            values = present_values if self.value == "present" else missing_values
            name = "Present" if self.value == "present" else "Missing"
            color = self.present_color if self.value == "present" else self.missing_color
            fig.add_bar(
                x=columns if self.orientation == "vertical" else values,
                y=values if self.orientation == "vertical" else columns,
                name=name,
                marker_color=color,
                text=[f"{int(v)}" if self.show_values else None for v in values],
                textposition="auto" if self.show_values else None,
            )

        first_col = columns[0] if columns else ""
        if self.orientation == "vertical":
            fig.update_xaxes(tickangle=-45, title_text=first_col)
            fig.update_yaxes(title_text="Count")
        else:
            fig.update_xaxes(title_text="Count")
            fig.update_yaxes(title_text=first_col)
        self._apply_base_layout(fig)
        return fig
