import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Literal

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
except Exception as exc:  # pragma: no cover - runtime optional dependency
    linkage = None
    dendrogram = None
    squareform = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


class _Dendrogram(_Plot):
    """
    Dendrogram of missingness correlation between columns.
    """

    def __init__(
        self,
        data: MissingData,
        selected_columns: Optional[List[str]] = None,
        max_columns: int = 30,
        drop_constant_columns: bool = True,
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
        self.max_columns = max_columns
        self.drop_constant_columns = drop_constant_columns
        self.linkage_method = linkage_method
        self.use_abs_correlation = use_abs_correlation
        self.line_width = line_width
        self.line_color = line_color

    def _build_figure(self) -> go.Figure:
        if linkage is None or dendrogram is None or squareform is None:
            raise ImportError(
                "Dendrogram requires scipy. Install it with: pip install scipy"
            ) from _SCIPY_IMPORT_ERROR

        missing_matrix = self.data.mask_missing

        if self.selected_columns is not None:
            cols = [c for c in self.selected_columns if c in missing_matrix.columns]
            if not cols:
                raise ValueError("No selected_columns found in DataFrame.")
            missing_matrix = missing_matrix[cols]

        if self.drop_constant_columns:
            constant_cols = [
                c for c in missing_matrix.columns
                if missing_matrix[c].nunique() <= 1
            ]
            if constant_cols:
                missing_matrix = missing_matrix.drop(columns=constant_cols)

        if self.max_columns > 0 and missing_matrix.shape[1] > self.max_columns:
            missing_matrix = missing_matrix.iloc[:, : self.max_columns]

        if missing_matrix.shape[1] < 2:
            raise ValueError(
                "Not enough columns with varying missingness to compute dendrogram."
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
        dendro = dendrogram(
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
            ),
            yaxis=dict(),
        )

        self._apply_base_layout(fig)
        return fig
