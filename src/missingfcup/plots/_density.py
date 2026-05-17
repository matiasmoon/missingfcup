import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convert a hex or rgb color string to an rgba string."""
    if color.startswith("rgba("):
        return color
    if color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f", {alpha})")
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(c * 2 for c in color)
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


class _Density(_Plot):
    """
    Overlapping KDE density curves split by missingness.

    Plots two density estimates for column ``x``:

    * **!NA** rows where ``color_by`` is *present*
    * **NA**  rows where ``color_by`` is *missing*

    If the two distributions overlap heavily → consistent with MCAR.
    If they diverge → missingness of ``color_by`` is associated with
    the values of ``x``, suggesting MAR or MNAR.
    """

    def __init__(
        self,
        data: MissingData,
        x: str,
        color_by: str,
        *,
        n_points: int = 300,
        fill_opacity: float = 0.4,
        **kwargs,
    ):
        legend_title = kwargs.pop("legend_title", f"{color_by}_NA")
        super().__init__(data=data, legend_title=legend_title, **kwargs)
        self.x = x
        self.color_by = color_by
        self.n_points = n_points
        self.fill_opacity = fill_opacity

    def _kde(self, values: np.ndarray, x_range: np.ndarray) -> np.ndarray:
        """KDE over x_range using scipy if available, else histogram interpolation."""
        try:
            from scipy.stats import gaussian_kde
            return gaussian_kde(values)(x_range)
        except ImportError:
            counts, edges = np.histogram(values, bins=50, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            return np.interp(x_range, centers, counts, left=0.0, right=0.0)

    def _build_figure(self) -> go.Figure:
        df = self.data.data

        if self.x not in df.columns:
            raise ValueError(f"Column '{self.x}' not found.")
        if self.color_by not in df.columns:
            raise ValueError(f"Column '{self.color_by}' not found.")
        if not pd.api.types.is_numeric_dtype(df[self.x]):
            raise TypeError(
                f"density() requires a numeric x column.\n"
                f"Column '{self.x}' has dtype '{df[self.x].dtype}'."
            )

        target_missing = self.data.mask_missing[self.color_by]
        x_all = df[self.x].dropna()
        span = (x_all.max() - x_all.min()) or 1.0
        x_range = np.linspace(
            x_all.min() - span * 0.05,
            x_all.max() + span * 0.05,
            self.n_points,
        )

        groups = [
            (~target_missing, "!NA", self.present_color),
            (target_missing,  "NA",  self.missing_color),
        ]

        fig = go.Figure()

        for mask, name, color in groups:
            vals = df.loc[mask, self.x].dropna().values
            if len(vals) < 2:
                continue
            density = self._kde(vals, x_range)
            fig.add_scatter(
                x=x_range,
                y=density,
                mode="lines",
                name=name,
                fill="tozeroy",
                fillcolor=_hex_to_rgba(color, self.fill_opacity),
                line=dict(color="black", width=1.5),
            )

        fig.update_xaxes(title_text=self.x)
        fig.update_yaxes(title_text="Density")
        fig.update_layout(dragmode="pan")
        self._apply_base_layout(fig)
        return fig
