from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots._plot import _Plot


class _AssociationHeatmap(_Plot):
    """Shared machinery for the three association heatmaps: correlation, predictive,
    and biserial.

    They all draw a matrix of association values in [-1, 1] and build the figure the same
    way (optional upper-triangle mask, in-cell value text, a grey underlay for NaN cells,
    and one main heatmap trace). A subclass only supplies:

    * ``_matrix()``          the DataFrame of values to draw,
    * ``_colorbar_config()`` the colorbar dict,
    * ``_hover_template()``  the hover string,

    and may override ``_apply_axes()`` to add axis titles.
    """

    def __init__(
        self,
        data: MissingData,
        *,
        selected_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.9,
        colorscale: str = "RdBu",
        show_values: bool = True,
        max_columns: int = 0,
        drop_constant_columns: bool = True,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 1,
        show_colorbar: bool = True,
        show_upper_triangle: bool = False,
        nan_color: str = "#c7c7c7",
        text_font_size: int = 12,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.ignore_high_missingness = ignore_high_missingness
        self.high_missingness_threshold = high_missingness_threshold
        self.colorscale = colorscale
        self.show_values = show_values
        self.max_columns = max_columns
        self.drop_constant_columns = drop_constant_columns
        self.order_by_missingness = order_by_missingness
        self.order = order
        self.value_round = value_round
        self.show_colorbar = show_colorbar
        self.show_upper_triangle = show_upper_triangle
        self.nan_color = nan_color
        self.text_font_size = text_font_size

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _matrix(self) -> pd.DataFrame:
        raise NotImplementedError

    def _colorbar_config(self) -> dict:
        raise NotImplementedError

    def _hover_template(self) -> str:
        raise NotImplementedError

    def _apply_axes(self, fig: go.Figure) -> None:
        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(tickangle=0, title_standoff=15)

    # ------------------------------------------------------------------
    # Shared figure construction
    # ------------------------------------------------------------------
    def _cell_text(self, corr_values: np.ndarray) -> np.ndarray:
        text = np.empty(corr_values.shape, dtype=object)
        rounded = np.round(corr_values, self.value_round)
        for idx in np.ndindex(text.shape):
            raw = corr_values[idx]
            r = rounded[idx]
            if np.isnan(r) or np.isclose(r, 0.0):
                text[idx] = ""
            elif np.isclose(abs(r), 1.0) and not np.isclose(abs(raw), 1.0):
                text[idx] = "<1" if r > 0 else ">-1"
            else:
                fmt = f"{r:.{self.value_round}f}"
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

        x_labels = corr.columns.tolist()
        y_labels = corr.index.tolist()
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
                    colorscale=[[0, self.nan_color], [1, self.nan_color]],
                    showscale=False,
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Heatmap(
                z=corr_values,
                x=x_labels,
                y=y_labels,
                colorscale=self.colorscale,
                zmin=-1,
                zmax=1,
                zmid=0,
                text=text if effective_show_values else None,
                texttemplate="%{text}" if effective_show_values else None,
                textfont=dict(size=self.text_font_size) if effective_show_values else None,
                showscale=self.show_colorbar,
                colorbar=self._colorbar_config() if self.show_colorbar else None,
                hovertemplate=self._hover_template(),
                hoverongaps=False,
            )
        )

        self._apply_axes(fig)
        self._apply_base_layout(fig)
        return fig
