import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Literal, List, Dict

from .Plot import Plot
from ..core.MissingData import MissingData

class Heatmap(Plot):
    """
    Interactive missingness heatmap.

    Rows = observations
    Columns = variables
    Cell color indicates missing vs present
    """

    def __init__(
        self,
        data: MissingData,
        *,
        selected_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.95,
        order_by: Optional[List[Dict]] = None,
        show_colorscale: bool = False,
        group_by_mode: Literal["binary", "missing"] = "binary",
        max_rows: Optional[int] = 500,
        **kwargs,
    ):
        super().__init__(
            data=data,
            legend_title="Status" if show_colorscale else None,
            **kwargs,
        )

        self.selected_columns = selected_columns
        self.ignore_high_missingness = ignore_high_missingness
        self.high_missingness_threshold = high_missingness_threshold
        self.order_by = order_by

        self.show_colorscale = show_colorscale
        self.group_by_mode = group_by_mode
        self.max_rows = max_rows

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _prepare_data(self) -> pd.DataFrame:
        df = self.data._filter_and_order(
            selected_columns=self.selected_columns,
            ignore_high_missingness=self.ignore_high_missingness,
            high_missingness_threshold=self.high_missingness_threshold,
            order_by=self.order_by,
        )

        # Row limiting (important for performance)
        if self.max_rows and len(df) > self.max_rows:
            df = df.iloc[: self.max_rows]

        return df

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        df = self._prepare_data()

        mask = df.isna()

        if self.group_by_mode == "binary":
            z = (~mask).astype(int).to_numpy()
            colorscale = [
                [0.0, self.missing_color],
                [1.0, self.present_color],
            ]
            colorbar_ticks = ["Missing", "Present"]
        else:
            z = mask.astype(int).to_numpy()
            colorscale = [
                [0.0, self.present_color],
                [1.0, self.missing_color],
            ]
            colorbar_ticks = ["Present", "Missing"]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=df.columns.tolist(),
                y=[str(i) for i in df.index],
                colorscale=colorscale,
                zmin=0,
                zmax=1,
                showscale=self.show_colorscale,
                colorbar=dict(
                    tickvals=[0, 1],
                    ticktext=colorbar_ticks,
                    len=0.4,
                )
                if self.show_colorscale
                else None,
            )
        )

        # Axis labels
        fig.update_layout(
            xaxis_title="Variables",
            yaxis_title="Rows",
        )

        # Apply shared layout/theme
        self._apply_base_layout(fig)

        return fig

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def fig(self) -> go.Figure:
        if self._figure is None:
            self._figure = self._build_figure()
        return self._figure

    def show(self):
        self.fig.show()

    def save(self, path: str):
        self.fig.write_html(path)
