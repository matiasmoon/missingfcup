import plotly.graph_objects as go
from typing import Optional, Literal, List, Dict
import pandas as pd

from .Plot import Plot
from ..core.MissingData import MissingData


class BarChart(Plot):
    """
    Bar chart summarizing missingness per column.
    """

    def __init__(
        self,
        data: MissingData,
        selected_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.9,
        completeness_mode: Optional[Literal["most", "least"]] = None,
        completeness_threshold: float = 0.0,
        max_columns_by_completeness: int = 0,
        max_columns: int = 50,
        order_by: Optional[List[Dict]] = None,
        mode: Literal["count", "percentage"] = "count",
        orientation: Literal["vertical", "horizontal"] = "vertical",
        stacked: bool = False,
        threshold: Optional[float] = None,
        threshold_color: str = "#ff7f0e",
        show_values: bool = True,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.ignore_high_missingness = ignore_high_missingness
        self.high_missingness_threshold = high_missingness_threshold
        self.completeness_mode = completeness_mode
        self.completeness_threshold = completeness_threshold
        self.max_columns_by_completeness = max_columns_by_completeness
        self.max_columns = max_columns
        self.order_by = order_by

        self.mode = mode
        self.orientation = orientation
        self.stacked = stacked

        self.threshold = threshold
        self.threshold_color = threshold_color
        self.show_values = show_values

    # ------------------------------------------------------------------
    # Data prep
    # ------------------------------------------------------------------
    def _prepare_data(self) -> pd.DataFrame:
        return self.data._filter_and_order(
            selected_columns=self.selected_columns,
            ignore_high_missingness=self.ignore_high_missingness,
            high_missingness_threshold=self.high_missingness_threshold,
            completeness_mode=self.completeness_mode,
            completeness_threshold=self.completeness_threshold,
            max_columns_by_completeness=self.max_columns_by_completeness,
            max_columns=self.max_columns,
            order_by=self.order_by,
        )

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        df = self._prepare_data()
        total_rows = len(df)

        missing_counts = df.isna().sum()
        present_counts = total_rows - missing_counts

        if self.mode == "percentage":
            missing_values = (missing_counts / total_rows) * 100
            present_values = 100 - missing_values
            value_suffix = "%"
            value_title = "% Missing"
        else:
            missing_values = missing_counts
            present_values = present_counts
            value_suffix = ""
            value_title = "Count of Missing Values"

        columns = missing_values.index.tolist()

        colors = (
            [
                self.threshold_color if v >= self.threshold else self.missing_color
                for v in missing_values
            ]
            if self.threshold is not None
            else self.missing_color
        )

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
            )

        # Axis titles
        fig.update_layout(
            xaxis_title="Column" if self.orientation == "vertical" else value_title,
            yaxis_title=value_title if self.orientation == "vertical" else "Column",
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
