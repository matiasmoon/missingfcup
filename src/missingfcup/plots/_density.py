import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
from missingfcup.plots._color import hex_to_rgba
from missingfcup.plots._plot import _Plot

# How finely the KDE is sampled for drawing. It sets resolution, not shape:
# the curve is the same continuous function either way, and above a few
# hundred points a denser grid only makes the figure heavier.
_CURVE_POINTS = 300

# Low enough that the two curves stay distinguishable where they overlap.
_FILL_OPACITY = 0.4


class _Density(_Plot):
    """
    Overlapping KDE density curves split by missingness.

    Plots two density estimates for column ``column``:

    * **!NA** rows where ``missing_column`` is *present*
    * **NA**  rows where ``missing_column`` is *missing*

    If the two distributions overlap heavily, that is consistent with MCAR.
    If they diverge, the missingness of ``missing_column`` is associated with
    the values of ``column``, suggesting MAR or MNAR.
    """

    def __init__(
        self,
        data: MissingData,
        column: str,
        missing_column: str,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)
        self.column = column
        self.missing_column = missing_column

    def _kde(self, values: np.ndarray, x_range: np.ndarray) -> np.ndarray:
        """Gaussian kernel density estimate of ``values`` evaluated over ``x_range``.

        Falls back to a histogram when the group has no spread: a constant column gives
        a singular covariance matrix, which ``gaussian_kde`` cannot invert.
        """
        try:
            return gaussian_kde(values)(x_range)
        except np.linalg.LinAlgError:
            counts, edges = np.histogram(values, bins=50, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            return np.interp(x_range, centers, counts, left=0.0, right=0.0)

    def _build_figure(self) -> go.Figure:
        df = self.data.data

        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found.")
        if self.missing_column not in df.columns:
            raise ValueError(f"Column '{self.missing_column}' not found.")
        if not pd.api.types.is_numeric_dtype(df[self.column]):
            raise TypeError(
                f"density() requires a numeric column column.\n"
                f"Column '{self.column}' has dtype '{df[self.column].dtype}'."
            )

        target_missing = self.data.mask_missing[self.missing_column]
        x_all = df[self.column].dropna()
        span = (x_all.max() - x_all.min()) or 1.0
        x_range = np.linspace(
            x_all.min() - span * 0.05,
            x_all.max() + span * 0.05,
            _CURVE_POINTS,
        )

        groups = [
            (~target_missing, f"!NA-{self.missing_column}", self.present_color),
            (target_missing, f"NA-{self.missing_column}", self.missing_color),
        ]

        fig = go.Figure()

        total = len(df)
        for mask, name, color in groups:
            vals = df.loc[mask, self.column].dropna().values
            if len(vals) < 2:
                continue
            density = self._kde(vals, x_range)
            fig.add_scatter(
                x=x_range,
                y=density,
                mode="lines",
                name=name,
                fill="tozeroy",
                fillcolor=hex_to_rgba(color, _FILL_OPACITY),
                line=dict(color="black", width=1.5),
                # The y-axis is deliberately unlabelled, so the hover is the only
                # place a reader can find out how much data is under a curve.
                hovertemplate=_hover.build(
                    _hover.title(f"{self.column}: %{{x:,.4~g}}"),
                    name,
                    _hover.rows_of_total(len(vals), total),
                ),
            )

        fig.update_xaxes(title_text=self.column)  # groups are named in the legend
        # A KDE integrates to 1 over the column range, so on a wide axis the values land
        # around 1e-5 and plotly renders them as "60u". The number carries nothing
        # for the reader: what matters is the shape and where the curves diverge.
        fig.update_yaxes(title_text="Density", showticklabels=False)
        fig.update_layout(dragmode="pan")
        self._apply_base_layout(fig)
        return fig
