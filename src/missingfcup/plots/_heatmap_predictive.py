import pandas as pd

from missingfcup.plots._association_heatmap import _AssociationHeatmap
from missingfcup.plots._selection import select_columns


class _HeatmapPredictive(_AssociationHeatmap):
    """
    Heatmap of present-vs-missing correlation.

    Each cell shows the correlation between being observed in one column
    versus missing in another (the present_missing mode).
    """

    def _matrix(self) -> pd.DataFrame:
        cols = select_columns(
            self.data,
            self.selected_columns,
            ignore_high_missingness=self.ignore_high_missingness,
            high_missingness_threshold=self.high_missingness_threshold,
            drop_constant=self.drop_constant_columns,
            order_by_missingness=self.order_by_missingness,
            order=self.order,
            max_columns=self.max_columns,
        )
        return self.data.present_missing_corr.loc[cols, cols]

    def _colorbar_config(self) -> dict:
        return dict(
            title=(
                "Present vs Missing"
                "<br><span style='font-size:10px'>blue = present predicts missing | red = both present</span>"
                "<br><span style='font-size:10px'>NaN = constant column</span>"
            ),
            tickmode="array",
            tickvals=[-1, 0, 1],
            ticktext=["Both present", "Independent", "Present predicts missing"],
        )

    def _hover_template(self) -> str:
        return (
            "<b>Present</b>: %{y}<br>"
            "<b>Missing</b>: %{x}<br>"
            "Correlation: %{z:.2f}<extra></extra>"
        )
