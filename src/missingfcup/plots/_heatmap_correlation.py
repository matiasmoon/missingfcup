import numpy as np
import pandas as pd

from missingfcup.plots._association_heatmap import _AssociationHeatmap
from missingfcup.plots._selection import select_columns


class _HeatmapCorrelation(_AssociationHeatmap):
    """
    Heatmap of column-level missingness correlations.

    Shows columns that tend to miss in the same rows (missing/missing correlation).
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
        if len(cols) < 2:
            raise ValueError(
                f"Only {len(cols)} column(s) have varying missingness, and a "
                f"correlation needs two. A column that is never missing, or always "
                f"missing, has no pattern to correlate."
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = self.data.mask_missing[cols].corr()
        return corr.loc[cols, cols]

    def _axis_roles(self) -> tuple:
        # Both axes are the same set of columns, so neither needs naming.
        return "", "", "Missingness correlation"

    def _axis_titles(self) -> tuple:
        # Symmetric: the same list of columns on both axes.
        return "Column", "Column"
