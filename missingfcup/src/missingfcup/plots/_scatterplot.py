import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData

class _ScatterPlot(_Plot):
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
        axis_padding: float = 0.1,
        missingness_color_column: Optional[str] = None,
        point_opacity: float = 0.7,
        jitter: float = 0.02,
        missing_jitter: float = 0.5,
        jitter_seed: int = 42,
        xaxis_range: Optional[list] = None,
        yaxis_range: Optional[list] = None,
        **kwargs,
    ):
        default_legend_title = (
            f"{missingness_color_column} missingness"
            if missingness_color_column is not None
            else "Status"
        )
        legend_title = kwargs.pop("legend_title", default_legend_title)
        super().__init__(data=data, legend_title=legend_title, **kwargs)
        self.x = x
        self.y = y
        self.point_size = point_size
        self.axis_padding = axis_padding
        self.missingness_color_column = missingness_color_column
        self.point_opacity = point_opacity
        self.jitter = jitter
        self.missing_jitter = missing_jitter
        self.jitter_seed = jitter_seed
        self.xaxis_range = xaxis_range
        self.yaxis_range = yaxis_range

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _prepare_df(self) -> pd.DataFrame:
        df = self.data.data
        if self.x not in df.columns or self.y not in df.columns:
            raise ValueError(f"Columns '{self.x}' and '{self.y}' must exist")
        if (
            self.missingness_color_column is not None
            and self.missingness_color_column not in df.columns
        ):
            raise ValueError(
                "missingness_color_column "
                f"'{self.missingness_color_column}' not found"
            )
        return df

    def _validate_numeric(self, df: pd.DataFrame) -> None:
        for col, axis in [(self.x, "x"), (self.y, "y")]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(
                    f"scatterplot_missingness() requires numeric columns.\n"
                    f"Column '{col}' has dtype '{df[col].dtype}'.\n"
                    f"Encode it first, e.g.:\n"
                    f"  df['{col}'] = pd.factorize(df['{col}'])[0]  # ordinal / nominal\n"
                    f"Or restrict to numeric columns only."
                )

    def _axis_tick_settings(self, series: pd.Series) -> dict:
        """Force integer ticks for columns whose observed values are all integers."""
        s = series.dropna()
        if s.empty:
            return {}
        if (s == s.round()).all():
            unique_vals = sorted(s.unique().tolist())
            if len(unique_vals) <= 20:
                return dict(
                    tickmode="array",
                    tickvals=unique_vals,
                    ticktext=[str(int(v)) for v in unique_vals],
                )
        return {}

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

    def _point_symbols(
        self,
        x_missing: pd.Series,
        y_missing: pd.Series,
        mask: pd.Series,
    ) -> np.ndarray:
        symbols = np.full(mask.sum(), "circle", dtype=object)
        selected_x_missing = x_missing[mask].to_numpy()
        selected_y_missing = y_missing[mask].to_numpy()

        symbols[selected_x_missing & ~selected_y_missing] = "x"
        symbols[~selected_x_missing & selected_y_missing] = "triangle-down"
        symbols[selected_x_missing & selected_y_missing] = "diamond-open"
        return symbols

    def _xy_status_labels(
        self,
        x_missing: pd.Series,
        y_missing: pd.Series,
        mask: pd.Series,
    ) -> np.ndarray:
        labels = np.full(mask.sum(), "Present", dtype=object)
        selected_x_missing = x_missing[mask].to_numpy()
        selected_y_missing = y_missing[mask].to_numpy()

        labels[selected_x_missing & ~selected_y_missing] = f"Missing {self.x}"
        labels[~selected_x_missing & selected_y_missing] = f"Missing {self.y}"
        labels[selected_x_missing & selected_y_missing] = "Missing both axes"
        return labels

    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()
        self._validate_numeric(df)

        x = df[self.x]
        y = df[self.y]

        missing_mask = self.data.mask_missing
        x_missing = missing_mask[self.x]
        y_missing = missing_mask[self.y]

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

        if self.missing_jitter > 0:
            rng = np.random.default_rng(self.jitter_seed + 1)
            x_span = (x.dropna().max() - x.dropna().min()) or 1.0
            y_span = (y.dropna().max() - y.dropna().min()) or 1.0
            jitter_x = self.missing_jitter * 0.05 * x_span
            jitter_y = self.missing_jitter * 0.05 * y_span
            if x_missing.any():
                plot_x = plot_x.copy()
                plot_x[x_missing] += rng.normal(0.0, jitter_x, size=int(x_missing.sum()))
            if y_missing.any():
                plot_y = plot_y.copy()
                plot_y[y_missing] += rng.normal(0.0, jitter_y, size=int(y_missing.sum()))

        both_present = ~x_missing & ~y_missing
        x_only_missing = x_missing & ~y_missing
        y_only_missing = ~x_missing & y_missing
        both_missing = x_missing & y_missing

        x_display = x.astype(object)
        y_display = y.astype(object)
        x_display[x_missing] = "Missing"
        y_display[y_missing] = "Missing"

        fig = go.Figure()

        if self.missingness_color_column is not None:
            target_missing = missing_mask[self.missingness_color_column]

            for is_missing, trace_name, trace_color in [
                (False, "Present", self.present_color),
                (True, "Missing", self.missing_color),
            ]:
                mask = target_missing == is_missing
                if not mask.any():
                    continue

                xy_status = self._xy_status_labels(x_missing, y_missing, mask)
                target_status = np.full(mask.sum(), trace_name, dtype=object)

                fig.add_scatter(
                    x=plot_x[mask],
                    y=plot_y[mask],
                    mode="markers",
                    name=trace_name,
                    marker=dict(
                        color=trace_color,
                        size=self.point_size,
                        symbol=self._point_symbols(x_missing, y_missing, mask),
                        opacity=self.point_opacity,
                    ),
                    customdata=np.column_stack(
                        [
                            x_display[mask].astype(object).to_numpy(),
                            y_display[mask].astype(object).to_numpy(),
                            target_status,
                            xy_status,
                        ]
                    ),
                    hovertemplate=(
                        f"<b>{self.x}</b>: %{{customdata[0]}}<br>"
                        f"<b>{self.y}</b>: %{{customdata[1]}}<br>"
                        f"<b>{self.missingness_color_column}</b>: %{{customdata[2]}}<br>"
                        "<b>Axes</b>: %{customdata[3]}<extra></extra>"
                    ),
                )

            fig.update_layout(dragmode="pan")

            def padded_range(series: pd.Series, offset_val: float) -> list[float]:
                s = series.dropna()
                if s.empty:
                    return [offset_val - 1, offset_val + 1]
                min_val = min(s.min(), offset_val)
                max_val = s.max()
                span = max_val - min_val or 1.0
                return [min_val - span * self.axis_padding, max_val + span * 0.15]

            fig.update_xaxes(range=self.xaxis_range or padded_range(x, x_offset), title_text=self.x, **self._axis_tick_settings(x))
            fig.update_yaxes(range=self.yaxis_range or padded_range(y, y_offset), title_text=self.y, **self._axis_tick_settings(y))
            self._apply_base_layout(fig)
            return fig

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

        fig.update_layout(dragmode="pan")

        def padded_range(series: pd.Series, offset_val: float) -> list[float]:
            s = series.dropna()
            if s.empty:
                return [offset_val - 1, offset_val + 1]

            min_val = min(s.min(), offset_val)
            max_val = s.max()
            span = max_val - min_val or 1.0
            pad = span * self.axis_padding

            return [min_val - pad, max_val + pad]

        fig.update_xaxes(range=self.xaxis_range or padded_range(x, x_offset), title_text=self.x)
        fig.update_yaxes(range=self.yaxis_range or padded_range(y, y_offset), title_text=self.y)

        self._apply_base_layout(fig)

        return fig
