import plotly.graph_objects as go
from typing import Literal

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData


class _BoxplotMissingness(_Plot):
    """
    Box or violin plot comparing the distribution of one column's values
    split by whether another column is missing or present.

    This is the primary visual tool for diagnosing MAR and MNAR:
    - If the two distributions look different → the missingness of ``color_by``
      may be related to the observed values of ``x`` (suggests MAR or MNAR).
    - If they look the same → no obvious relationship (consistent with MCAR).

    Parameters
    ----------
    x : str
        Column whose value distribution is shown on the y-axis.
    color_by : str
        Column whose missingness splits the two groups.
    plot_type : str
        ``"box"`` for box plots (simpler, shows quartiles and outliers).
        ``"violin"`` for violin plots (also shows the full distribution shape).

    Example
    -------
    md.boxplot_missingness(x="fare", color_by="age")
    → "Do passengers with missing age tend to pay different fares?"
    """

    def __init__(
        self,
        data: MissingData,
        x: str,
        color_by: str,
        *,
        plot_type: Literal["box", "violin"] = "box",
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)
        self.x = x
        self.color_by = color_by
        self.plot_type = plot_type

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        import pandas as pd
        df = self.data.data
        if self.x not in df.columns:
            raise ValueError(f"Column '{self.x}' not found in DataFrame.")
        if self.color_by not in df.columns:
            raise ValueError(f"Column '{self.color_by}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[self.x]):
            raise TypeError(
                f"boxplot_missingness() requires a numeric column for 'x'.\n"
                f"Column '{self.x}' has dtype '{df[self.x].dtype}'.\n"
                f"Encode it first, e.g.:\n"
                f"  df['{self.x}'] = pd.factorize(df['{self.x}'])[0]"
            )

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self) -> go.Figure:
        self._validate()

        values = self.data.data[self.x]
        is_missing = self.data.mask_missing[self.color_by]

        present_vals = values[~is_missing].dropna()
        missing_vals = values[is_missing].dropna()

        TraceClass = go.Violin if self.plot_type == "violin" else go.Box

        fig = go.Figure()

        fig.add_trace(TraceClass(
            y=present_vals,
            name=f"{self.color_by}: present",
            marker_color=self.present_color,
        ))

        fig.add_trace(TraceClass(
            y=missing_vals,
            name=f"{self.color_by}: missing",
            marker_color=self.missing_color,
        ))

        fig.update_layout(
        )

        self._apply_base_layout(fig)
        return fig
