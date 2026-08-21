from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
from missingfcup.plots._plot import _Plot

# Grey, because it marks a cell whose association could not be computed at all,
# not a value on the scale. It has to sit off the green-white-red ramp entirely:
# painted in missing_color an undefined cell would read as the strongest possible
# positive association, which is the one reading it must never have.
_NAN_COLOR = "#c7c7c7"


class _AssociationHeatmap(_Plot):
    """Shared machinery for the three association heatmaps: correlation, predictive,
    and biserial.

    They all draw a matrix of association values in [-1, 1] and build the figure the same
    way (optional upper-triangle mask, in-cell value text, a grey underlay for NaN cells,
    and one main heatmap trace). A subclass only supplies:

    * ``_matrix()``          the DataFrame of values to draw,
    * ``_hover_template()``  the hover string,

    and may override ``_apply_axes()`` to add axis titles.
    """

    def __init__(
        self,
        data: MissingData,
        *,
        selected_columns: Optional[List[str]] = None,
        high_missingness_threshold: Optional[float] = None,
        show_values: bool = True,
        max_columns: int = 0,
        drop_constant_columns: bool = False,
        sort_by: Optional[Literal["missingness", "alphabetical"]] = "missingness",
        ascending: bool = False,
        show_upper_triangle: bool = False,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.high_missingness_threshold = high_missingness_threshold
        self.show_values = show_values
        self.max_columns = max_columns
        self.drop_constant_columns = drop_constant_columns
        self.sort_by = sort_by
        self.ascending = ascending
        self.show_upper_triangle = show_upper_triangle

    def _matrix(self) -> pd.DataFrame:
        raise NotImplementedError

    def _colorbar_config(self) -> dict:
        """Compact bar, no title, and only the ends and midpoint marked.

        All three heatmaps share a -1..1 scale, so the labels between the ends add
        width without adding meaning.
        """
        return dict(
            tickvals=[-1, 0, 1],
            len=0.5,
            thickness=14,
        )

    def _axis_roles(self) -> tuple:
        """What the row axis and the column axis mean, and what the number is called.

        The three kinds differ only in this, so keeping it here means one tooltip
        shape across all of them rather than three that read differently.
        """
        raise NotImplementedError

    def _axis_titles(self) -> tuple:
        """(x, y) axis titles. Like the hover, the three kinds differ only here."""
        raise NotImplementedError

    def _hover_template(self) -> str:
        row_role, col_role, statistic = self._axis_roles()
        # A role is empty when the axis needs no explaining, as on the symmetric
        # kinds where both axes are the same set of columns.
        row = f"{row_role} %{{y}}".strip()
        col = f"{col_role} %{{x}}".strip()
        return _hover.build(
            _hover.title(f"{row}  \u2192  {col}"),
            f"{statistic}: %{{z:.2f}}",
        )

    def _apply_axes(self, fig: go.Figure) -> None:
        x_title, y_title = self._axis_titles()
        fig.update_xaxes(tickangle=-45, title_text=x_title)
        fig.update_yaxes(tickangle=0, title_text=y_title, title_standoff=15)

    def _cell_text(self, corr_values: np.ndarray) -> np.ndarray:
        text = np.empty(corr_values.shape, dtype=object)
        rounded = np.round(corr_values, 2)
        for idx in np.ndindex(text.shape):
            raw = corr_values[idx]
            r = rounded[idx]
            if np.isnan(r) or np.isclose(r, 0.0):
                text[idx] = ""
            elif np.isclose(abs(r), 1.0) and not np.isclose(abs(raw), 1.0):
                text[idx] = "<1" if r > 0 else ">-1"
            else:
                fmt = f"{r:.2f}"
                text[idx] = fmt.rstrip("0").rstrip(".")
        return text

    def _build_figure(self) -> go.Figure:
        corr = self._matrix()

        effective_show_values = self.show_values and max(corr.shape) <= 30

        # Hide the lower triangle when asked, for square matrices only.
        if self.show_upper_triangle and corr.shape[0] == corr.shape[1]:
            tri_mask = pd.DataFrame(
                np.tril(np.ones(corr.shape, dtype=bool), k=-1),
                index=corr.index,
                columns=corr.columns,
            )
            corr = corr.mask(tri_mask)

        x_labels = self._truncate_labels(corr.columns.tolist())
        y_labels = self._truncate_labels(corr.index.tolist())
        corr_values = corr.to_numpy(dtype=float)
        nan_mask = np.isnan(corr_values)

        text = self._cell_text(corr_values) if effective_show_values else None

        fig = go.Figure()

        if nan_mask.any():
            fig.add_trace(
                go.Heatmap(
                    z=np.where(nan_mask, 1.0, np.nan),
                    x=x_labels,
                    y=y_labels,
                    colorscale=[[0, _NAN_COLOR], [1, _NAN_COLOR]],
                    hovertemplate=_hover.build(
                        _hover.title("%{y}  \u2192  %{x}"),
                        "not defined",
                    ),
                    showscale=False,
                )
            )

        fig.add_trace(
            go.Heatmap(
                z=corr_values,
                x=x_labels,
                y=y_labels,
                # The poles of all three kinds are 'more missing' against 'more
                # present', so they use the same two colours as every other plot.
                colorscale=[
                    [0.0, self.present_color],
                    [0.5, "#ffffff"],
                    [1.0, self.missing_color],
                ],
                zmin=-1,
                zmax=1,
                zmid=0,
                text=text if effective_show_values else None,
                texttemplate="%{text}" if effective_show_values else None,
                showscale=self.show_legend,
                colorbar=self._colorbar_config() if self.show_legend else None,
                hovertemplate=self._hover_template(),
                hoverongaps=False,
            )
        )

        self._apply_axes(fig)
        self._apply_base_layout(fig)
        return fig
