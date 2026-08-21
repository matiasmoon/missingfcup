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
            high_missingness_threshold=self.high_missingness_threshold,
            drop_constant=self.drop_constant_columns,
            sort_by=self.sort_by,
            ascending=self.ascending,
            max_columns=self.max_columns,
        )
        return self.data.present_missing_corr.loc[cols, cols]

    def _axis_roles(self) -> tuple:
        return "presence of", "missingness of", "Correlation"

    def _axis_titles(self) -> tuple:
        return "Missing column", "Observed column"
