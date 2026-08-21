from typing import List, Literal, Optional

import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
from missingfcup.plots._plot import _Plot
from missingfcup.plots._selection import select_columns

# The rate strip is one row of cells, so the value written inside a cell has about
# width/n_columns pixels to live in. Past this many columns the numbers collide into
# an unreadable smear, and the colour plus the hover carry the reading instead. It is
# a legibility limit rather than a preference, so it is fixed rather than an option.
_MAX_COLUMNS_WITH_VALUES = 20


class _Rate(_Plot):
    """
    Heatmap showing missing rate per column.

    Single-row heatmap where each cell represents
    the fraction or percentage of missing values
    in a column.
    """

    def __init__(
        self,
        data: MissingData,
        *,
        selected_columns: Optional[List[str]] = None,
        high_missingness_threshold: Optional[float] = None,
        measure: Literal["fraction", "percentage"] = "fraction",
        show_values: bool = True,
        max_columns: int = 0,
        sort_by: Optional[Literal["missingness", "alphabetical"]] = "missingness",
        ascending: bool = False,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.high_missingness_threshold = high_missingness_threshold
        self.measure = measure
        self.show_values = show_values
        self.max_columns = max_columns
        self.sort_by = sort_by
        self.ascending = ascending

    def _build_figure(self) -> go.Figure:
        cols = select_columns(
            self.data,
            self.selected_columns,
            high_missingness_threshold=self.high_missingness_threshold,
            sort_by=self.sort_by,
            ascending=self.ascending,
            max_columns=self.max_columns,
        )
        if not cols:
            raise ValueError(
                f"No columns left to draw out of {len(self.data.columns)}. Check "
                f"high_missingness_threshold={self.high_missingness_threshold!r} and "
                f"max_columns={self.max_columns!r}."
            )
        rates = self.data.col_missing_rate.loc[cols]

        if self.measure == "percentage":
            values = rates * 100
            text = [[f"{v:.2f}%" for v in values]]
        else:
            values = rates
            text = [[f"{v:.2f}" for v in values]]

        labels_display = self._truncate_labels(values.index.tolist())

        zmin = 0
        zmax = max(values.max(), 1e-6)
        suffix = "%" if self.measure == "percentage" else ""

        show_cell_text = self.show_values and len(values) <= _MAX_COLUMNS_WITH_VALUES

        counts = self.data.col_missing_count.loc[values.index]
        total = len(self.data.data)
        # A template rather than pre-rendered strings, so this plot formats the same
        # way as every other one instead of having its own copy of the rules.
        customdata = [
            [
                (
                    name,
                    _hover.rate(val, self.measure == "percentage"),
                    _hover.rows_of_total(count, total),
                )
                for name, val, count in zip(values.index, values, counts)
            ]
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=[values.values],
                x=labels_display,
                y=["Missing rate"],
                # 0 is no missing, the maximum is the most missing, so the scale
                # runs from bare paper to the colour missingness is drawn in.
                colorscale=[[0.0, "#ffffff"], [1.0, self.missing_color]],
                zmin=zmin,
                zmax=zmax,
                xgap=1,
                ygap=1,
                text=text if show_cell_text else None,
                texttemplate="%{text}" if show_cell_text else None,
                showscale=self.show_legend,
                colorbar=dict(
                    # Only the ends: the value is written in every cell already.
                    tickvals=[zmin, zmax],
                    ticktext=[
                        f"{zmin:.2f}{suffix}",
                        f"{zmax:.2f}{suffix}",
                    ],
                    len=0.5,
                    thickness=14,
                )
                if self.show_legend
                else None,
                hovertemplate=_hover.build(
                    _hover.title("%{customdata[0]}"),
                    "NA: %{customdata[1]}",
                    "%{customdata[2]}",
                ),
                customdata=customdata,
            )
        )

        fig.update_layout(yaxis=dict(showticklabels=False))
        fig.update_xaxes(tickangle=-45, title_text="Column")
        fig.update_yaxes(title_standoff=15)

        self._apply_base_layout(fig)

        return fig
