import plotly.graph_objects as go
import pandas as pd
from typing import Optional

from .Plot import Plot
from ..core.MissingData import MissingData


class PatternBarChart(Plot):
    """
    Bar chart showing frequency of row-level missingness patterns.

    Each bar represents a unique combination of missing variables.
    Height corresponds to number of rows with that pattern.
    """

    def __init__(
        self,
        data: MissingData,
        bar_color: str = "#7f7f7f",
        max_patterns: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.bar_color = bar_color
        self.max_patterns = max_patterns

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        df = self.data.data

        # Compute row-level missingness patterns
        def row_pattern(row: pd.Series) -> str:
            missing = row.index[row.isna()].tolist()
            if not missing:
                return "No values missing"
            return ", ".join(missing)

        patterns = df.apply(row_pattern, axis=1)
        counts = patterns.value_counts()

        if self.max_patterns is not None:
            counts = counts.iloc[: self.max_patterns]

        fig = go.Figure()

        fig.add_bar(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            marker_color=self.bar_color,
            text=counts.values.tolist(),
            textposition="outside",
            hovertemplate=(
                "<b>Pattern</b>: %{x}<br>"
                "<b>Rows</b>: %{y}<extra></extra>"
            ),
        )

        fig.update_layout(
            xaxis_title="Missingness pattern",
            yaxis_title="Number of rows",
            xaxis_tickangle=-45,
            margin=dict(t=60, b=140),
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
