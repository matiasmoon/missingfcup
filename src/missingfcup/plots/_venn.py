from typing import List, Literal, Optional

import pandas as pd
import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots._plot import _Plot


class _Venn(_Plot):
    """
    Bar chart of the 7 exclusive missingness subsets for 3 columns.

    Each bar represents one of the 7 exclusive combinations:
    3 single-column patterns, 3 two-column patterns, and 1 three-column pattern:
    the bar-chart equivalent of a 3-set Venn diagram.
    """

    def __init__(
        self,
        data: MissingData,
        *,
        selected_columns: Optional[List[str]] = None,
        order: Literal["desc", "asc"] = "desc",
        value: Literal["count", "percent"] = "count",
        show_values: bool = True,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.order = order
        self.value = value
        self.show_values = show_values

    def _prepare_columns(self) -> List[str]:
        df = self.data.data
        if self.selected_columns:
            cols = [c for c in self.selected_columns if c in df.columns]
            if len(cols) < 3:
                raise ValueError("venn requires at least 3 valid columns.")
            return cols[:3]

        missing_rate = self.data.col_missing_rate
        cols = missing_rate.sort_values(ascending=False).head(3).index.tolist()
        if len(cols) < 3:
            raise ValueError("venn requires at least 3 columns.")
        return cols

    def _build_figure(self) -> go.Figure:
        cols = self._prepare_columns()
        df = self.data.data
        mask = self.data.mask_missing[cols]

        a, b, c = cols
        subsets = [
            (a,),
            (b,),
            (c,),
            (a, b),
            (a, c),
            (b, c),
            (a, b, c),
        ]

        def subset_count(subset: tuple[str, ...]) -> int:
            cond = pd.Series(True, index=df.index)
            for col in cols:
                if col in subset:
                    cond &= mask[col]
                else:
                    cond &= ~mask[col]
            return int(cond.sum())

        labels_full = [", ".join(s) for s in subsets]
        counts = [subset_count(s) for s in subsets]

        if self.order == "asc":
            ordered = sorted(zip(labels_full, counts), key=lambda pair: pair[1])
            labels_full = [label for label, _ in ordered]
            counts = [count for _, count in ordered]

        total_rows = len(df)
        if self.value == "percent":
            values = [c / max(total_rows, 1) * 100.0 for c in counts]
            y_title = "Percent of rows"
            text_values = (
                [f"{v:.1f}%" if v > 0 else "" for v in values] if self.show_values else None
            )
        else:
            values = [float(c) for c in counts]
            y_title = "Number of rows"
            text_values = (
                [f"{int(v)}" if v > 0 else "" for v in values] if self.show_values else None
            )

        labels_display = self._truncate_labels(labels_full)

        fig = go.Figure()
        fig.add_bar(
            x=labels_display,
            y=values,
            name="Missing values",
            marker_color=self.missing_color,
            text=text_values,
            textposition="outside" if self.show_values else None,
            hovertemplate=(
                "<b>Missing columns</b>: %{customdata[2]}<br>"
                "<b>Rows</b>: %{customdata[1]}<br>"
                "<b>Percent</b>: %{customdata[0]:.1f}%<extra></extra>"
            ),
            customdata=[
                [percent, count, full]
                for percent, count, full in zip(
                    [c / max(total_rows, 1) * 100.0 for c in counts],
                    counts,
                    labels_full,
                )
            ],
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            margin=dict(t=90, b=140),
            bargap=0.25,
            yaxis=dict(rangemode="tozero"),
            uniformtext=dict(minsize=8, mode="hide"),
        )
        fig.update_traces(textangle=0, textfont=dict(size=10), cliponaxis=False)
        fig.update_xaxes(title_text="Missing columns")
        fig.update_yaxes(automargin=True, rangemode="tozero", title_text=y_title)

        max_val = max(values) if values else 0
        if max_val > 0:
            fig.update_yaxes(range=[0, max_val * 1.3])

        self._apply_base_layout(fig)
        return fig
