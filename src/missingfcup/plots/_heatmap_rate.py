import plotly.graph_objects as go
import pandas as pd
from typing import Optional, List, Literal

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData

class _HeatmapRate(_Plot):
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
        rates = self.data.col_missing_rate

        if self.ignore_high_missingness:
            rates = rates[rates < self.high_missingness_threshold]

        if self.selected_columns is not None:
            cols = [c for c in self.selected_columns if c in rates.index]
            if not cols:
                raise ValueError("No selected_columns found in DataFrame.")
            rates = rates.loc[cols]

        if rates.empty:
            raise ValueError("No columns available to plot")

        if self.order_by_missingness:
            rates = rates.sort_values(ascending=self.order == "asc")

        if self.max_columns > 0 and len(rates) > self.max_columns:
            rates = rates.iloc[: self.max_columns]

        if self.scale == "percentage":
            values = rates * 100
            label = "Missing (%)"
            text = [[f"{v:.{self.value_round}f}%" for v in values]]
        else:
            values = rates
            label = "Missing rate"
            text = [[f"{v:.{self.value_round}f}" for v in values]]

        def resolved_max_label_length() -> int:
            if self.max_label_length > 0:
                return self.max_label_length
            return max(16, int(self.width / 12))

        max_len = resolved_max_label_length()

        def truncate_label(label: str) -> str:
            if max_len <= 0 or len(label) <= max_len:
                return label
            return label[: max_len - 1] + "…"

        labels_display = [truncate_label(l) for l in values.index.tolist()]

        if len(set(labels_display)) < len(labels_display):
            counts_seen = {}
            adjusted = []
            for lbl in labels_display:
                counts_seen[lbl] = counts_seen.get(lbl, 0) + 1
                idx = counts_seen[lbl]
                suffix = " ..."
                base = lbl
                if max_len > len(suffix):
                    base = lbl[: max_len - len(suffix)]
                adjusted.append(base + suffix + (" " * (idx - 1)))
            labels_display = adjusted

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
