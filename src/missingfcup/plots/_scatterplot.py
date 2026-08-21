from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
from missingfcup.plots._plot import _Plot

# Fixed presentation: these were options nobody needed to tune, and the plot is
# designed around these values.
_POINT_SIZE = 8
_POINT_OPACITY = 0.7

# A missing value has no coordinate, so it is parked this far below the observed
# minimum, as a fraction of the observed span. The gap is what separates "missing"
# from "small" at a glance.
_OFFSET_GAP = 0.1

# Jitter on the offset band is capped so about three standard deviations still fit
# inside that gap. Without the cap a large jitter would push missing points up into
# the observed range, where nothing distinguishes them from real values.
_MISSING_JITTER_CAP = _OFFSET_GAP / 3


class _Scatterplot(_Plot):
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
        *,
        axis_padding: float = 0.1,
        missing_column: Optional[str] = None,
        jitter: float = 0.02,
        jitter_seed: int = 42,
        xaxis_range: Optional[list] = None,
        yaxis_range: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)
        self.x = x
        self.y = y
        self.axis_padding = axis_padding
        self.missing_column = missing_column
        self.jitter = jitter
        self.jitter_seed = jitter_seed
        self.xaxis_range = xaxis_range
        self.yaxis_range = yaxis_range

    def _prepare_df(self) -> pd.DataFrame:
        df = self.data.data
        if self.x not in df.columns or self.y not in df.columns:
            absent = [c for c in (self.x, self.y) if c not in df.columns]
            raise ValueError(
                f"Columns not found in the DataFrame: {absent}. It holds {list(df.columns)}."
            )
        if self.missing_column is not None and self.missing_column not in df.columns:
            raise ValueError(f"missing_column '{self.missing_column}' not found")
        return df

    def _validate_numeric(self, df: pd.DataFrame) -> None:
        for col in (self.x, self.y):
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(
                    f"scatterplot() requires numeric columns.\n"
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

    def _padded_range(
        self, series: pd.Series, offset_val: float, *, upper_pad: Optional[float] = None
    ) -> List[float]:
        """Axis range wide enough to show both the data and the offset missing markers.

        ``upper_pad`` overrides the padding above the data; the coloured variant leaves
        extra headroom there for the legend.
        """
        s = series.dropna()
        if s.empty:
            return [offset_val - 1, offset_val + 1]
        min_val = min(s.min(), offset_val)
        max_val = s.max()
        span = max_val - min_val or 1.0
        top = span * (self.axis_padding if upper_pad is None else upper_pad)
        return [min_val - span * self.axis_padding, max_val + top]

    def _compute_offset(self, series: pd.Series) -> float:
        s = series.dropna()
        if s.empty:
            return 0.0
        span = s.max() - s.min() or 1.0
        return s.min() - _OFFSET_GAP * span

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
        labels = np.full(mask.sum(), "!NA", dtype=object)
        selected_x_missing = x_missing[mask].to_numpy()
        selected_y_missing = y_missing[mask].to_numpy()

        labels[selected_x_missing & ~selected_y_missing] = f"NA-{self.x}"
        labels[~selected_x_missing & selected_y_missing] = f"NA-{self.y}"
        labels[selected_x_missing & selected_y_missing] = "NA-both"
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
            # One pass over every point. Missing rows all share the offset
            # coordinate, so they need the spread at least as much as tied observed
            # values do -- but theirs is capped to stay inside the offset gap.
            rng = np.random.default_rng(self.jitter_seed)
            x_span = (x.dropna().max() - x.dropna().min()) or 1.0
            y_span = (y.dropna().max() - y.dropna().min()) or 1.0
            capped = min(self.jitter, _MISSING_JITTER_CAP)

            plot_x = plot_x + rng.normal(
                0.0, np.where(x_missing, capped, self.jitter) * x_span, size=len(plot_x)
            )
            plot_y = plot_y + rng.normal(
                0.0, np.where(y_missing, capped, self.jitter) * y_span, size=len(plot_y)
            )

        both_present = ~x_missing & ~y_missing
        x_only_missing = x_missing & ~y_missing
        y_only_missing = ~x_missing & y_missing
        both_missing = x_missing & y_missing

        x_display = x.astype(object)
        y_display = y.astype(object)
        x_display[x_missing] = "NA"
        y_display[y_missing] = "NA"

        fig = go.Figure()

        if self.missing_column is not None:
            target_missing = missing_mask[self.missing_column]

            for is_missing, trace_name, trace_color in [
                (False, f"!NA-{self.missing_column}", self.present_color),
                (True, f"NA-{self.missing_column}", self.missing_color),
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
                        size=_POINT_SIZE,
                        symbol=self._point_symbols(x_missing, y_missing, mask),
                        opacity=_POINT_OPACITY,
                    ),
                    customdata=np.column_stack(
                        [
                            x_display[mask].astype(object).to_numpy(),
                            y_display[mask].astype(object).to_numpy(),
                            target_status,
                            xy_status,
                        ]
                    ),
                    hovertemplate=_hover.build(
                        _hover.title(
                            f"{self.x}: %{{customdata[0]}} \u00b7 {self.y}: %{{customdata[1]}}"
                        ),
                        "%{customdata[3]}",
                        f"{self.missing_column}: %{{customdata[2]}}",
                    ),
                )

            fig.update_layout(dragmode="pan")

            def padded_range(series, offset_val):
                return self._padded_range(series, offset_val, upper_pad=0.15)

            fig.update_xaxes(
                range=self.xaxis_range or padded_range(x, x_offset),
                title_text=self.x,
                **self._axis_tick_settings(x),
            )
            fig.update_yaxes(
                range=self.yaxis_range or padded_range(y, y_offset),
                title_text=self.y,
                **self._axis_tick_settings(y),
            )
            self._apply_base_layout(fig)
            return fig

        if both_present.any():
            fig.add_scatter(
                x=plot_x[both_present],
                y=plot_y[both_present],
                mode="markers",
                name="!NA",
                marker=dict(
                    color=self.present_color,
                    size=_POINT_SIZE,
                    symbol="circle",
                    opacity=_POINT_OPACITY,
                ),
                customdata=self._make_customdata(x_display, y_display, both_present),
                hovertemplate=_hover.build(
                    _hover.title(
                        f"{self.x}: %{{customdata[0]}} \u00b7 {self.y}: %{{customdata[1]}}"
                    ),
                    "!NA",
                ),
            )

        if x_only_missing.any():
            fig.add_scatter(
                x=plot_x[x_only_missing],
                y=plot_y[x_only_missing],
                mode="markers",
                name=f"NA-{self.x}",
                marker=dict(
                    color=self.missing_color,
                    size=_POINT_SIZE,
                    symbol="x",
                    opacity=_POINT_OPACITY,
                ),
                customdata=self._make_customdata(x_display, y_display, x_only_missing),
                hovertemplate=_hover.build(
                    _hover.title(
                        f"{self.x}: %{{customdata[0]}} \u00b7 {self.y}: %{{customdata[1]}}"
                    ),
                    f"NA-{self.x}",
                ),
            )

        if y_only_missing.any():
            fig.add_scatter(
                x=plot_x[y_only_missing],
                y=plot_y[y_only_missing],
                mode="markers",
                name=f"NA-{self.y}",
                marker=dict(
                    color=self.missing_color,
                    size=_POINT_SIZE,
                    symbol="triangle-down",
                    opacity=_POINT_OPACITY,
                ),
                customdata=self._make_customdata(x_display, y_display, y_only_missing),
                hovertemplate=_hover.build(
                    _hover.title(
                        f"{self.x}: %{{customdata[0]}} \u00b7 {self.y}: %{{customdata[1]}}"
                    ),
                    f"NA-{self.y}",
                ),
            )

        if both_missing.any():
            fig.add_scatter(
                x=plot_x[both_missing],
                y=plot_y[both_missing],
                mode="markers",
                name="NA-both",
                marker=dict(
                    color=self.missing_color,
                    size=_POINT_SIZE,
                    symbol="diamond-open",
                    opacity=_POINT_OPACITY,
                ),
                customdata=self._make_customdata(x_display, y_display, both_missing),
                hovertemplate=_hover.build(
                    _hover.title(
                        f"{self.x}: %{{customdata[0]}} \u00b7 {self.y}: %{{customdata[1]}}"
                    ),
                    "NA-both",
                ),
            )

        fig.update_layout(dragmode="pan")

        fig.update_xaxes(
            range=self.xaxis_range or self._padded_range(x, x_offset), title_text=self.x
        )
        fig.update_yaxes(
            range=self.yaxis_range or self._padded_range(y, y_offset), title_text=self.y
        )

        self._apply_base_layout(fig)

        return fig
