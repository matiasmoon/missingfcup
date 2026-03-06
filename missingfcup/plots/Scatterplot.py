import plotly.graph_objects as go
import pandas as pd
import numpy as np

from missingfcup.plots.Plot import Plot
from missingfcup.core.MissingData import MissingData

class ScatterPlot(Plot):
    """
    Scatter plot of two columns, highlighting missingness.

    Points are grouped by whether x and/or y are missing.
    Missing values are visualized using axis offsets.
    """

    def __init__(
        self,
        data: MissingData,
        x: str,
        y: str,
        point_size: int = 8,
        axis_padding: float = 0.05,
        point_opacity: float = 0.7,
        jitter: float = 0.0,
        jitter_seed: int = 42,
        **kwargs,
    ):
        legend_title = kwargs.pop("legend_title", "Status")
        super().__init__(data=data, legend_title=legend_title, **kwargs)
        self.x = x
        self.y = y
        self.point_size = point_size
        self.axis_padding = axis_padding
        self.point_opacity = point_opacity
        self.jitter = jitter
        self.jitter_seed = jitter_seed

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _prepare_df(self) -> pd.DataFrame:
        df = self.data.data
        if self.x not in df.columns or self.y not in df.columns:
            raise ValueError(f"Columns '{self.x}' and '{self.y}' must exist")
        return df

    def _validate_numeric(self, df: pd.DataFrame) -> None:
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

    def _compute_offset(self, series: pd.Series) -> float:
        s = series.dropna()
        if s.empty:
            return 0.0
        span = s.max() - s.min() or 1.0
        return s.min() - 0.1 * span

    def _make_customdata(
        self,
        x_display: pd.Series,
        y_display: pd.Series,
        mask: pd.Series,
    ) -> np.ndarray:
        return np.column_stack(
            [
                x_display[mask].astype(object).to_numpy(),
                y_display[mask].astype(object).to_numpy(),
            ]
        )

    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()
        self._validate_numeric(df)

        x = df[self.x]
        y = df[self.y]

        # Reuse cached missingness info
        missing_mask = self.data.missing_mask
        x_missing = missing_mask[self.x]
        y_missing = missing_mask[self.y]

        # ------------------------------------------------------------------
        # Offset computation (for missing visualization)
        # ------------------------------------------------------------------
        x_offset = self._compute_offset(x)
        y_offset = self._compute_offset(y)

        plot_x = x.copy()
        plot_y = y.copy()

        plot_x[x_missing] = x_offset
        plot_y[y_missing] = y_offset

        if self.jitter > 0:
            rng = np.random.default_rng(self.jitter_seed)
            x_span = (x.dropna().max() - x.dropna().min()) or 1.0
            y_span = (y.dropna().max() - y.dropna().min()) or 1.0
            plot_x = plot_x + rng.normal(0.0, self.jitter * x_span, size=len(plot_x))
            plot_y = plot_y + rng.normal(0.0, self.jitter * y_span, size=len(plot_y))

        both_present = ~x_missing & ~y_missing
        x_only_missing = x_missing & ~y_missing
        y_only_missing = ~x_missing & y_missing
        both_missing = x_missing & y_missing

        x_display = x.astype(object)
        y_display = y.astype(object)
        x_display[x_missing] = "Missing"
        y_display[y_missing] = "Missing"

        fig = go.Figure()

        # ------------------------------------------------------------------
        # Present points
        # ------------------------------------------------------------------
        if both_present.any():
            fig.add_scatter(
                x=plot_x[both_present],
                y=plot_y[both_present],
                mode="markers",
                name="Present",
                marker=dict(
                    color=self.present_color,
                    size=self.point_size,
                    symbol="circle",
                    opacity=self.point_opacity,
                ),
                customdata=self._make_customdata(x_display, y_display, both_present),
                hovertemplate=(
                    f"<b>{self.x}</b>: %{{customdata[0]}}<br>"
                    f"<b>{self.y}</b>: %{{customdata[1]}}<br>"
                    "<b>Status</b>: Present<extra></extra>"
                ),
            )

        # ------------------------------------------------------------------
        # Missing points (x only)
        # ------------------------------------------------------------------
        if x_only_missing.any():
            fig.add_scatter(
                x=plot_x[x_only_missing],
                y=plot_y[x_only_missing],
                mode="markers",
                name=f"Missing {self.x}",
                marker=dict(
                    color=self.missing_color,
                    size=self.point_size,
                    symbol="x",
                    opacity=self.point_opacity,
                ),
                customdata=self._make_customdata(
                    x_display, y_display, x_only_missing
                ),
                hovertemplate=(
                    f"<b>{self.x}</b>: %{{customdata[0]}}<br>"
                    f"<b>{self.y}</b>: %{{customdata[1]}}<br>"
                    f"<b>Status</b>: Missing {self.x}<extra></extra>"
                ),
            )

        # ------------------------------------------------------------------
        # Missing points (y only)
        # ------------------------------------------------------------------
        if y_only_missing.any():
            fig.add_scatter(
                x=plot_x[y_only_missing],
                y=plot_y[y_only_missing],
                mode="markers",
                name=f"Missing {self.y}",
                marker=dict(
                    color=self.missing_color,
                    size=self.point_size,
                    symbol="triangle-down",
                    opacity=self.point_opacity,
                ),
                customdata=self._make_customdata(
                    x_display, y_display, y_only_missing
                ),
                hovertemplate=(
                    f"<b>{self.x}</b>: %{{customdata[0]}}<br>"
                    f"<b>{self.y}</b>: %{{customdata[1]}}<br>"
                    f"<b>Status</b>: Missing {self.y}<extra></extra>"
                ),
            )

        # ------------------------------------------------------------------
        # Missing points (both)
        # ------------------------------------------------------------------
        if both_missing.any():
            fig.add_scatter(
                x=plot_x[both_missing],
                y=plot_y[both_missing],
                mode="markers",
                name="Missing Both",
                marker=dict(
                    color=self.missing_color,
                    size=self.point_size,
                    symbol="diamond-open",
                    opacity=self.point_opacity,
                ),
                customdata=self._make_customdata(x_display, y_display, both_missing),
                hovertemplate=(
                    f"<b>{self.x}</b>: %{{customdata[0]}}<br>"
                    f"<b>{self.y}</b>: %{{customdata[1]}}<br>"
                    "<b>Status</b>: Missing both<extra></extra>"
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

    # Public API inherited from Plot
