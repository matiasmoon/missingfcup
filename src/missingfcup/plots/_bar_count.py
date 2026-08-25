import pandas as pd
import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
from missingfcup.plots._bar_base import _BarBase


class _BarCount(_BarBase):
    """
    Bar chart of missing and/or present counts per column.
    """

    def __init__(
        self,
        data: MissingData,
        *,
        show_both: bool = False,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)
        self.show_both = show_both

    def _compute_missing_values(self, df: pd.DataFrame) -> pd.Series:
        return self.data.col_missing_count.loc[df.columns]

    def _compute_present_values(self, df: pd.DataFrame) -> pd.Series:
        return self.data.col_present_count.loc[df.columns]

    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()
        missing_values = self._compute_missing_values(df)
        present_values = self._compute_present_values(df)
        full_names = missing_values.index.tolist()
        columns = self._truncate_labels(full_names)
        total = len(self.data.data)

        def hover(values, state):
            # Truncated axis label may be an ellipsis, so the tooltip carries the
            # full name: the one place with room for it.
            return dict(
                customdata=_hover.customdata(
                    full_names, [_hover.rows_of_total(v, total) for v in values]
                ),
                hovertemplate=_hover.build(
                    _hover.title("%{customdata[0]}"), f"{state}: %{{customdata[1]}}"
                ),
            )

        fig = go.Figure()

        if self.show_both:
            fig.add_bar(
                x=columns if self.orientation == "vertical" else missing_values,
                y=missing_values if self.orientation == "vertical" else columns,
                name="NA",
                marker_color=self.missing_color,
                text=[f"{int(v)}" if self.show_values else None for v in missing_values],
                textposition="auto" if self.show_values else None,
                **hover(missing_values, "NA"),
            )
            fig.add_bar(
                x=columns if self.orientation == "vertical" else present_values,
                y=present_values if self.orientation == "vertical" else columns,
                name="!NA",
                marker_color=self.present_color,
                text=[f"{int(v)}" if self.show_values else None for v in present_values],
                textposition="auto" if self.show_values else None,
                **hover(present_values, "!NA"),
            )
            fig.update_layout(barmode="stack")
        else:
            # Missing count is the whole point of the plot; show_both is how a
            # caller asks to see the present side too.
            values = missing_values
            # One series means the legend would repeat what the title already says.
            fig.add_bar(
                showlegend=False,
                x=columns if self.orientation == "vertical" else values,
                y=values if self.orientation == "vertical" else columns,
                name="NA",
                marker_color=self.missing_color,
                text=[f"{int(v)}" if self.show_values else None for v in values],
                textposition="auto" if self.show_values else None,
                **hover(values, "NA"),
            )

        # Stacked, a bar's height is every row; alone, it is only the missing ones.
        value_title = "Rows" if self.show_both else "Missing rows"
        if self.orientation == "vertical":
            fig.update_xaxes(tickangle=-45, title_text="Column")
            fig.update_yaxes(title_text=value_title)
        else:
            fig.update_xaxes(title_text=value_title)
            fig.update_yaxes(title_text="Column")
        self._apply_base_layout(fig)
        return fig
