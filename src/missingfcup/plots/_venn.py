from typing import List, Literal, Optional

import pandas as pd
import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
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
        sort_by: Optional[Literal["size"]] = "size",
        ascending: bool = False,
        measure: Literal["count", "fraction", "percentage"] = "count",
        show_values: bool = True,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.sort_by = sort_by
        self.ascending = ascending
        self.measure = measure
        self.show_values = show_values

    def _prepare_columns(self) -> List[str]:
        """The three columns to compare, which the caller has to name.

        A three-set Venn has exactly 7 exclusive regions, so the plot is defined by
        three columns and no other number will do. Choosing them here -- by missing
        rate, say -- would mean the figure silently answered a different question
        from the one that was asked.
        """
        if not self.selected_columns:
            # Naming them is required, but the caller should not have to go looking:
            # suggest the three with the most missing data.
            suggestion = (
                self.data.col_missing_rate.loc[lambda s: s > 0]
                .sort_values(ascending=False)
                .head(3)
                .index.tolist()
            )
            hint = f" The 3 emptiest are {suggestion}." if len(suggestion) == 3 else ""
            raise ValueError(f"venn() needs selected_columns: name the 3 columns to compare.{hint}")
        df = self.data.data
        missing = [c for c in self.selected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in the DataFrame: {missing}.")
        if len(self.selected_columns) != 3:
            raise ValueError(
                f"venn() compares exactly 3 columns, got {len(self.selected_columns)}. "
                "Use upset() to compare a different number."
            )
        return list(self.selected_columns)

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

        if self.ascending:
            ordered = sorted(zip(labels_full, counts), key=lambda pair: pair[1])
            labels_full = [label for label, _ in ordered]
            counts = [count for _, count in ordered]

        total_rows = len(df)
        # measure means the same here as on bar() and rate(): one option covering
        # absolute counts and the two relative forms.
        if self.measure != "count":
            rates = [c / max(total_rows, 1) for c in counts]
            if self.measure == "percentage":
                values = [r * 100.0 for r in rates]
                y_title = "Percent of rows"
                fmt = "{:.2f}%"
            else:
                values = rates
                y_title = "Fraction of rows"
                fmt = "{:.2f}"
            text_values = (
                [fmt.format(v) if v > 0 else "" for v in values] if self.show_values else None
            )
        else:
            values = [float(c) for c in counts]
            y_title = "Number of rows"
            text_values = (
                [f"{int(v)}" if v > 0 else "" for v in values] if self.show_values else None
            )

        labels_display = self._truncate_labels(labels_full)

        fig = go.Figure()
        # Only ever one series here too.
        fig.add_bar(
            showlegend=False,
            x=labels_display,
            y=values,
            name="NA",
            marker_color=self.missing_color,
            text=text_values,
            textposition="outside" if self.show_values else None,
            # Count and share regardless of measure: the tooltip has room for both,
            # so it should not make the reader change measure to see the other one.
            hovertemplate=_hover.build(
                _hover.title("%{customdata[2]}"),
                "%{customdata[1]}",
            ),
            customdata=[
                [percent, count, full]
                for percent, count, full in zip(
                    [c / max(total_rows, 1) * 100.0 for c in counts],
                    [_hover.rows_of_total(c, total_rows) for c in counts],
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
        fig.update_traces(textangle=0, cliponaxis=False)
        fig.update_xaxes(title_text="Missing columns")
        fig.update_yaxes(automargin=True, rangemode="tozero", title_text=y_title)

        max_val = max(values) if values else 0
        if max_val > 0:
            fig.update_yaxes(range=[0, max_val * 1.3])

        self._apply_base_layout(fig)
        return fig
