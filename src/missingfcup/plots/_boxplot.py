from typing import Literal

import pandas as pd
import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
from missingfcup.plots._plot import _Plot


class _Boxplot(_Plot):
    """
    Box or violin plot comparing the distribution of one column's values
    split by whether another column is missing or present.

    This is the primary visual tool for diagnosing MAR and MNAR:
    * If the two distributions look different, the missingness of ``missing_column``
      may be related to the observed values of ``column`` (suggests MAR or MNAR).
    * If they look the same, there is no obvious relationship (consistent with MCAR).

    Parameters
    ----------
    column : str
        Column whose value distribution is shown on the y-axis.
    missing_column : str
        Column whose missingness splits the two groups.
    shape : str
        ``"box"`` for box plots (simpler, shows quartiles and outliers).
        ``"violin"`` for violin plots (also shows the full distribution shape).

    Example
    -------
    md.boxplot(column="fare", missing_column="age")
    answers "Do passengers with missing age tend to pay different fares?"
    """

    def __init__(
        self,
        data: MissingData,
        column: str,
        missing_column: str,
        *,
        shape: Literal["box", "violin"] = "box",
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)
        self.column = column
        self.missing_column = missing_column
        self.shape = shape

    def _validate(self) -> None:
        df = self.data.data
        if self.column not in df.columns:
            raise ValueError(f"Column {self.column!r} not found in the DataFrame.")
        if self.missing_column not in df.columns:
            raise ValueError(f"Column {self.missing_column!r} not found in the DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[self.column]):
            raise TypeError(
                f"boxplot() requires a numeric column for 'column'.\n"
                f"Column '{self.column}' has dtype '{df[self.column].dtype}'.\n"
                f"Encode it first, e.g.:\n"
                f"  df['{self.column}'] = pd.factorize(df['{self.column}'])[0]"
            )

    def _build_figure(self) -> go.Figure:
        self._validate()

        values = self.data.data[self.column]
        is_missing = self.data.mask_missing[self.missing_column]

        present_vals = values[~is_missing].dropna()
        missing_vals = values[is_missing].dropna()

        TraceClass = go.Violin if self.shape == "violin" else go.Box

        fig = go.Figure()

        total = len(values)
        for group_vals, name, color in [
            (present_vals, f"!NA-{self.missing_column}", self.present_color),
            (missing_vals, f"NA-{self.missing_column}", self.missing_color),
        ]:
            fig.add_trace(
                TraceClass(
                    y=group_vals,
                    name=name,
                    marker_color=color,
                    # Box hides its own sample size, the first thing to check
                    # before reading anything into the spread.
                    hovertemplate=_hover.build(
                        _hover.title(f"{self.column}: %{{y:,.4~g}}"),
                        name,
                        _hover.rows_of_total(len(group_vals), total),
                    ),
                )
            )

        fig.update_xaxes(tickangle=-45, title_text=f"{self.missing_column} missingness")
        fig.update_yaxes(title_text=self.column)

        self._apply_base_layout(fig)
        return fig
