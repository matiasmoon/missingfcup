import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Literal

from .Plot import Plot
from ..core.MissingData import MissingData
from ..core.ViewMetadata import ViewMetadata, OrderType, NumericOrder


class BarChart(Plot):
    """
    Bar chart summarizing missingness per column.

    Supports counts or percentages, vertical or horizontal orientation,
    optional stacking (present vs missing), ordering via metadata,
    and threshold-based highlighting.
    """

    def __init__(
        self,
        data: MissingData,
        metadata: Optional[ViewMetadata] = None,
        figure_size_pixels: tuple[int, int] = (900, 500),
        mode: Literal["count", "percentage"] = "count",
        orientation: Literal["vertical", "horizontal"] = "vertical",
        stacked: bool = False,
        missing_color: str = "#d62728",
        present_color: str = "#2ca02c",
        threshold: Optional[float] = None,
        threshold_color: str = "#ff7f0e",
        show_values: bool = True,
        title: Optional[str] = None,
    ):
        self.data = data
        self.metadata = metadata

        self.width, self.height = figure_size_pixels
        self.mode = mode
        self.orientation = orientation
        self.stacked = stacked

        self.missing_color = missing_color
        self.present_color = present_color

        self.threshold = threshold
        self.threshold_color = threshold_color
        self.show_values = show_values
        self.title = title

        self._figure: Optional[go.Figure] = None

    # ------------------------------------------------------------------
    # Ordering logic (shared with Heatmap)
    # ------------------------------------------------------------------
    def _apply_ordering(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.metadata is None or not self.metadata.order_by:
            return df

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

        total_rows = len(df)
        missing_counts = df.isna().sum()
        present_counts = total_rows - missing_counts

        if self.mode == "percentage":
            missing_values = (missing_counts / total_rows) * 100
            present_values = 100 - missing_values
            y_title = "% Missing"
            value_suffix = "%"
        else:
            missing_values = missing_counts
            present_values = present_counts
            y_title = "Count of Missing Values"
            value_suffix = ""

        columns = missing_values.index.tolist()

        # Threshold-based coloring
        if self.threshold is not None:
            colors = [
                self.threshold_color if v >= self.threshold else self.missing_color
                for v in missing_values
            ]
        else:
            colors = self.missing_color

        fig = go.Figure()

        if self.stacked:
            fig.add_bar(
                x=columns if self.orientation == "vertical" else missing_values,
                y=missing_values if self.orientation == "vertical" else columns,
                name="Missing",
                marker_color=self.missing_color,
            )
            fig.add_bar(
                x=columns if self.orientation == "vertical" else present_values,
                y=present_values if self.orientation == "vertical" else columns,
                name="Present",
                marker_color=self.present_color,
            )
            fig.update_layout(barmode="stack")
        else:
            fig.add_bar(
                x=columns if self.orientation == "vertical" else missing_values,
                y=missing_values if self.orientation == "vertical" else columns,
                marker_color=colors,
                text=[
                    f"{v:.1f}{value_suffix}" if self.show_values else None
                    for v in missing_values
                ],
                textposition="auto" if self.show_values else None,
                hovertemplate=(
                    "<b>Column</b>: %{x}<br>"
                    f"<b>Missing</b>: %{{y:.1f}}{value_suffix}<extra></extra>"
                )
                if self.orientation == "vertical"
                else (
                    "<b>Column</b>: %{y}<br>"
                    f"<b>Missing</b>: %{{x:.1f}}{value_suffix}<extra></extra>"
                ),
            )

        fig.update_layout(
            title=self.title or "Missing Values per Column",
            xaxis_title="Column" if self.orientation == "vertical" else y_title,
            yaxis_title=y_title if self.orientation == "vertical" else "Column",
            width=self.width,
            height=self.height,
            showlegend=self.stacked,
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