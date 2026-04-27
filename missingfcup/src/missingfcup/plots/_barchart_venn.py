import plotly.graph_objects as go
import pandas as pd
from typing import Optional, List, Literal

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData


class _BarchartVenn(_Plot):
    """
    Bar chart of the 7 exclusive missingness subsets for 3 columns.

    Each bar represents one of the 7 exclusive combinations:
    3 single-column patterns, 3 two-column patterns, and 1 three-column pattern —
    the bar-chart equivalent of a 3-set Venn diagram.
    """

    def __init__(
        self,
        data: MissingData,
        selected_columns: Optional[List[str]] = None,
        sort_order: Literal["desc", "asc"] = "desc",
        value: Literal["count", "percent"] = "count",
        show_values: bool = True,
        max_label_length: int = 48,
        missing_color: str = "#d62728",
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.sort_order = sort_order
        self.value = value
        self.show_values = show_values
        self.max_label_length = max_label_length
        self.missing_color = missing_color

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _prepare_columns(self) -> List[str]:
        df = self.data.data
        if self.selected_columns:
            cols = [c for c in self.selected_columns if c in df.columns]
            if len(cols) < 3:
                raise ValueError("barchart_venn requires at least 3 valid columns.")
            return cols[:3]

        missing_rate = self.data.col_missing_rate
        cols = missing_rate.sort_values(ascending=False).head(3).index.tolist()
        if len(cols) < 3:
            raise ValueError("barchart_venn requires at least 3 columns.")
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

        if self.sort_order == "asc":
            labels_full, counts = zip(
                *sorted(zip(labels_full, counts), key=lambda x: x[1])
            )
            labels_full = list(labels_full)
            counts = list(counts)

        total_rows = len(df)
        if self.value == "percent":
            values = [c / max(total_rows, 1) * 100.0 for c in counts]
            y_title = "Percent of rows"
            text_values = [
                f"{v:.1f}%" if v > 0 else ""
                for v in values
            ] if self.show_values else None
        else:
            values = counts
            y_title = "Number of rows"
            text_values = [
                f"{int(v)}" if v > 0 else ""
                for v in values
            ] if self.show_values else None

        def resolved_max_label_length() -> int:
            if self.max_label_length > 0:
                return self.max_label_length
            return max(16, int(self.width / 12))

        max_len = resolved_max_label_length()

        def truncate_label(label: str) -> str:
            if max_len <= 0 or len(label) <= max_len:
                return label
            return label[: max_len - 1] + "…"

        labels_display = [truncate_label(l) for l in labels_full]

        if len(set(labels_display)) < len(labels_display):
            counts_seen = {}
            adjusted = []
            for label in labels_display:
                counts_seen[label] = counts_seen.get(label, 0) + 1
                idx = counts_seen[label]
                suffix = " ..."
                base = label
                if max_len > len(suffix):
                    base = label[: max_len - len(suffix)]
                adjusted.append(base + suffix + (" " * (idx - 1)))
            labels_display = adjusted

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
        fig.update_xaxes(title_text=a)
        fig.update_yaxes(automargin=True, rangemode="tozero", title_text=y_title)

        max_val = max(values) if values else 0
        if max_val > 0:
            fig.update_yaxes(range=[0, max_val * 1.3])

        self._apply_base_layout(fig)
        return fig
