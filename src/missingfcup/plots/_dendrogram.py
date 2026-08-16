from typing import List, Literal, Optional

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram as _scipy_dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from missingfcup.core.missing_data import MissingData
from missingfcup.plots._plot import _Plot
from missingfcup.plots._selection import select_columns


class _Dendrogram(_Plot):
    """
    Dendrogram of missingness correlation between columns.
    """

    def __init__(
        self,
        data: MissingData,
        *,
        selected_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.9,
        max_columns: int = 30,
        drop_constant_columns: bool = False,
        linkage_method: Literal[
            "single", "complete", "average", "weighted", "centroid", "median", "ward"
        ] = "average",
        use_abs_correlation: bool = False,
        line_width: int = 2,
        line_color: str = "#4C78A8",
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.ignore_high_missingness = ignore_high_missingness
        self.high_missingness_threshold = high_missingness_threshold
        self.max_columns = max_columns
        self.drop_constant_columns = drop_constant_columns
        self.linkage_method = linkage_method
        self.use_abs_correlation = use_abs_correlation
        self.line_width = line_width
        self.line_color = line_color

    def _build_figure(self) -> go.Figure:
        cols = select_columns(
            self.data,
            self.selected_columns,
            ignore_high_missingness=self.ignore_high_missingness,
            high_missingness_threshold=self.high_missingness_threshold,
            drop_constant=self.drop_constant_columns,
            max_columns=self.max_columns,
        )
        missing_matrix = self.data.mask_missing[cols]

        if missing_matrix.shape[1] < 2:
            raise ValueError(
                "Not enough columns with varying missingness to compute the dendrogram."
            )

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = missing_matrix.corr()
        corr = corr.fillna(0.0)

        if self.use_abs_correlation:
            corr = corr.abs()

        distance = 1.0 - corr
        np.fill_diagonal(distance.values, 0.0)

        condensed = squareform(distance.values, checks=False)
        linkage_matrix = linkage(condensed, method=self.linkage_method)
        dendro = _scipy_dendrogram(
            linkage_matrix,
            labels=list(distance.columns),
            no_plot=True,
        )

        fig = go.Figure()
        for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=self.line_color, width=self.line_width),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        labels = dendro["ivl"]
        leaf_positions = list(range(5, 10 * len(labels) + 5, 10))

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=leaf_positions,
                ticktext=labels,
                tickangle=-45,
                title="Column",
            ),
            yaxis=dict(
                title="Distance",
            ),
        )

        self._apply_base_layout(fig)
        return fig
