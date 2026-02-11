import plotly.graph_objects as go
import pandas as pd

from .Plot import Plot
from ..core.MissingData import MissingData

class ScatterPlot(Plot):
    """
    Scatter plot of two columns, highlighting missingness.

    A point is considered missing if either x or y is missing.
    Missing values are visualized using axis offsets.
    """

    def __init__(
        self,
        data: MissingData,
        x: str,
        y: str,
        point_size: int = 8,
        axis_padding: float = 0.05,
        **kwargs,
    ):
        super().__init__(
            data=data,
            legend_title="Status",
            **kwargs,
        )
        self.x = x
        self.y = y
        self.point_size = point_size
        self.axis_padding = axis_padding

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        df = self.data.data

        if self.x not in df.columns or self.y not in df.columns:
            raise ValueError(f"Columns '{self.x}' and '{self.y}' must exist")

        x = df[self.x]
        y = df[self.y]

        # Ensure numeric data
        if not pd.api.types.is_numeric_dtype(df[self.x]):
            raise TypeError(
                f"ScatterPlot requires numeric x-axis. "
                f"Column '{self.x}' has dtype {df[self.x].dtype}."
            )

        if not pd.api.types.is_numeric_dtype(df[self.y]):
            raise TypeError(
                f"ScatterPlot requires numeric y-axis. "
                f"Column '{self.y}' has dtype {df[self.y].dtype}."
            )

        # Reuse cached missingness info
        missing_matrix = self.data.missing_matrix
        x_missing = missing_matrix[self.x]
        y_missing = missing_matrix[self.y]
        any_missing = x_missing | y_missing

        # ------------------------------------------------------------------
        # Offset computation (for missing visualization)
        # ------------------------------------------------------------------
        def compute_offset(series: pd.Series) -> float:
            s = series.dropna()
            if s.empty:
                return 0.0
            span = s.max() - s.min() or 1.0
            return s.min() - 0.1 * span

        x_offset = compute_offset(x)
        y_offset = compute_offset(y)

        plot_x = x.copy()
        plot_y = y.copy()

        plot_x[x_missing] = x_offset
        plot_y[y_missing] = y_offset

        fig = go.Figure()

        # ------------------------------------------------------------------
        # Present points
        # ------------------------------------------------------------------
        fig.add_scatter(
            x=plot_x[~any_missing],
            y=plot_y[~any_missing],
            mode="markers",
            name="Present",
            marker=dict(
                color=self.present_color,
                size=self.point_size,
            ),
            hovertemplate=(
                f"<b>{self.x}</b>: %{{x}}<br>"
                f"<b>{self.y}</b>: %{{y}}<extra></extra>"
            ),
        )

        # ------------------------------------------------------------------
        # Missing points
        # ------------------------------------------------------------------
        fig.add_scatter(
            x=plot_x[any_missing],
            y=plot_y[any_missing],
            mode="markers",
            name="Missing",
            marker=dict(
                color=self.missing_color,
                size=self.point_size,
                symbol="circle-open",
            ),
            hovertemplate=(
                f"<b>{self.x}</b>: %{{x}}<br>"
                f"<b>{self.y}</b>: %{{y}}<br>"
                "<b>Status</b>: Missing<extra></extra>"
            ),
        )

        # ------------------------------------------------------------------
        # Axis labels & interaction
        # ------------------------------------------------------------------
        fig.update_layout(
            xaxis_title=self.x,
            yaxis_title=self.y,
            dragmode="pan",
        )

        # ------------------------------------------------------------------
        # Padded axis ranges (default zoom-out)
        # ------------------------------------------------------------------
        def padded_range(series: pd.Series, offset_val: float) -> list[float]:
            s = series.dropna()
            if s.empty:
                return [offset_val - 1, offset_val + 1]

            min_val = min(s.min(), offset_val)
            max_val = s.max()
            span = max_val - min_val or 1.0
            pad = span * self.axis_padding

            return [min_val - pad, max_val + pad]

        fig.update_xaxes(range=padded_range(x, x_offset))
        fig.update_yaxes(range=padded_range(y, y_offset))

        # ------------------------------------------------------------------
        # Shared layout/theme
        # ------------------------------------------------------------------
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
