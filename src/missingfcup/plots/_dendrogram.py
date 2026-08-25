from typing import List, Literal, Optional

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram as _scipy_dendrogram
from scipy.cluster.hierarchy import linkage as _scipy_linkage
from scipy.spatial.distance import squareform

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
from missingfcup.plots._plot import _Plot
from missingfcup.plots._selection import select_columns

# Tree is the whole plot, so its lines are drawn heavy enough to read as
# structure, not annotation.
_LINE_WIDTH = 3.0

# Tree carries no missing/present meaning -- structure, not data -- so it is drawn
# in a neutral colour that stays out of the package's red/green vocabulary.
_LINE_COLOR = "#4C78A8"


class _Dendrogram(_Plot):
    """
    Dendrogram of missingness correlation between columns.
    """

    def __init__(
        self,
        data: MissingData,
        *,
        selected_columns: Optional[List[str]] = None,
        high_missingness_threshold: Optional[float] = None,
        max_columns: int = 0,
        drop_constant_columns: bool = False,
        linkage: Literal[
            "single", "complete", "average", "weighted", "centroid", "median", "ward"
        ] = "average",
        use_abs_correlation: bool = False,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.high_missingness_threshold = high_missingness_threshold
        self.max_columns = max_columns
        self.drop_constant_columns = drop_constant_columns
        self.linkage = linkage
        self.use_abs_correlation = use_abs_correlation

    def _build_figure(self) -> go.Figure:
        cols = select_columns(
            self.data,
            self.selected_columns,
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
        linkage_matrix = _scipy_linkage(condensed, method=self.linkage)
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
                    line=dict(color=_LINE_COLOR, width=_LINE_WIDTH),
                    # Height of a join is the only number this plot encodes and no
                    # axis tick lands on it, so without a tooltip it is unreadable.
                    hovertemplate=_hover.build(
                        _hover.title("Merge"),
                        "Distance: %{y:.2f}",
                    ),
                    showlegend=False,
                )
            )

        labels = self._truncate_labels(list(dendro["ivl"]))
        leaf_positions = list(range(5, 10 * len(labels) + 5, 10))

        # scipy places leaves at 5, 15, 25, ... so the first and last sit half a
        # step from the data edge. Pad by that same half step, leave headroom
        # above the tallest join.
        half_step = 5
        max_height = max((y for ys in dendro["dcoord"] for y in ys), default=1.0)

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=leaf_positions,
                ticktext=labels,
                tickangle=-45,
                title="Column",
                range=[leaf_positions[0] - half_step, leaf_positions[-1] + half_step],
            ),
            yaxis=dict(
                title="Distance",
                range=[0, max_height * 1.08],
            ),
        )

        self._apply_base_layout(fig)
        return fig
