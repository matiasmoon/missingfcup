import plotly.graph_objects as go
import pandas as pd
from typing import Optional, List, Literal

from missingfcup.plots._plot import _Plot
from missingfcup.plots._selection import select_columns
from missingfcup.core.missing_data import MissingData

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
        selected_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.9,
        scale: Literal["fraction", "percentage"] = "fraction",
        colorscale: str = "Reds",
        show_values: bool = True,
        max_columns: int = 30,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 2,
        show_colorbar: bool = True,
        max_labels_with_values: int = 20,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.ignore_high_missingness = ignore_high_missingness
        self.high_missingness_threshold = high_missingness_threshold
        self.scale = scale
        self.colorscale = colorscale
        self.show_values = show_values
        self.max_columns = max_columns
        self.order_by_missingness = order_by_missingness
        self.order = order
        self.value_round = value_round
        self.show_colorbar = show_colorbar
        self.max_labels_with_values = max_labels_with_values

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        cols = select_columns(
            self.data,
            self.selected_columns,
            ignore_high_missingness=self.ignore_high_missingness,
            high_missingness_threshold=self.high_missingness_threshold,
            order_by_missingness=self.order_by_missingness,
            order=self.order,
            max_columns=self.max_columns,
        )
        if not cols:
            raise ValueError("No columns available to plot")
        rates = self.data.col_missing_rate.loc[cols]

        if self.scale == "percentage":
            values = rates * 100
            label = "Missing (%)"
            text = [[f"{v:.{self.value_round}f}%" for v in values]]
        else:
            values = rates
            label = "Missing rate"
            text = [[f"{v:.{self.value_round}f}" for v in values]]

        labels_display = self._truncate_labels(values.index.tolist())

        zmin = 0
        zmax = max(values.max(), 1e-6)

        show_cell_text = self.show_values and len(values) <= self.max_labels_with_values

        customdata = [
            [
                (
                    name,
                    f"{val:.{self.value_round}f}%" if self.scale == "percentage"
                    else f"{val:.{self.value_round}f}"
                )
                for name, val in zip(values.index, values)
            ]
        ]

        hovertext = [
            [
                f"<b>Column</b>: {name}<br><b>{label}</b>: "
                + (
                    f"{val:.{self.value_round}f}%"
                    if self.scale == "percentage"
                    else f"{val:.{self.value_round}f}"
                )
                for name, val in zip(values.index, values)
            ]
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=[values.values],
                x=labels_display,
                y=["Missing rate"],
                colorscale=self.colorscale,
                zmin=zmin,
                zmax=zmax,
                xgap=1,
                ygap=1,
                text=text if show_cell_text else None,
                texttemplate="%{text}" if show_cell_text else None,
                showscale=self.show_colorbar,
                colorbar=dict(title=label) if self.show_colorbar else None,
                hovertext=hovertext,
                hovertemplate="%{hovertext}<extra></extra>",
                customdata=customdata,
            )
        )

        fig.update_layout(yaxis=dict(showticklabels=False))
        first_col = values.index[0] if len(values) > 0 else ""
        fig.update_xaxes(tickangle=-45, title_text=first_col)
        fig.update_yaxes(title_standoff=15)

        self._apply_base_layout(fig)

        return fig
