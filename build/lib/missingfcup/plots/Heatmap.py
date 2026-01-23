import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Literal

from .Plot import Plot
from ..core.MissingData import MissingData
from ..core.ViewMetadata import ViewMetadata, OrderType, NumericOrder


class Heatmap(Plot):
    """
    Interactive missingness heatmap visualization.

    Displays a grid where each cell represents whether data is present or missing.
    Rows correspond to records, columns correspond to variables.

    Ordering and grouping logic is handled via metadata, keeping visualization
    concerns separate from data logic.
    """

    def __init__(
        self,
        data: MissingData,
        metadata: Optional[ViewMetadata] = None,
        figure_size_pixels: tuple[int, int] = (900, 600),
        present_color: str = "#2ca02c",
        missing_color: str = "#d62728",
        show_colorscale_legend: bool = False,
        title: Optional[str] = None,
        group_by_mode: Literal["binary", "completeness"] = "binary",
    ):
        self.data = data
        self.metadata = metadata

        self.width, self.height = figure_size_pixels
        self.present_color = present_color
        self.missing_color = missing_color
        self.show_scale = show_colorscale_legend
        self.title = title
        self.group_by_mode = group_by_mode

        self._figure: Optional[go.Figure] = None

    # ------------------------------------------------------------------
    # Ordering logic (delegated via metadata)
    # ------------------------------------------------------------------
    def _apply_ordering(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.metadata is None or not self.metadata.order_by:
            return df

        # Apply ordering specs in reverse for stable multi-column sorting
        for spec in reversed(self.metadata.order_by):
            if spec.type == OrderType.NUMERIC:
                df = df.sort_values(
                    spec.column,
                    ascending=(spec.numeric_order == NumericOrder.ASC),
                    kind="stable",
                )
            else:
                cat = pd.Categorical(
                    df[spec.column],
                    categories=spec.category_order,
                    ordered=True,
                )
                df = df.assign(**{spec.column: cat}).sort_values(
                    spec.column,
                    kind="stable",
                )

        return df

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        df = self._apply_ordering(self.data.data)

        # Missingness mask
        mask = df.isna()
        values = (~mask).astype(int).to_numpy()

        # Visualization mode
        if self.group_by_mode == "binary":
            z = values
            colorscale = [[0, self.missing_color], [1, self.present_color]]
        else:
            z = 1 - values
            colorscale = [[0, self.present_color], [1, self.missing_color]]

        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=df.columns.tolist(),
                y=[str(i) for i in df.index],
                colorscale=colorscale,
                showscale=self.show_scale,
                zmin=0,
                zmax=1,
                colorbar=dict(
                    tickmode="array",
                    tickvals=[0, 1],
                    ticktext=["Missing", "Present"]
                    if self.group_by_mode == "binary"
                    else ["0%", "100%"],
                    len=0.3,
                )
                if self.show_scale
                else None,
            )
        )

        fig.update_layout(
            title=self.title or "Missingness Heatmap",
            width=self.width,
            height=self.height,
            xaxis_title="",
            yaxis_title="",
        )

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