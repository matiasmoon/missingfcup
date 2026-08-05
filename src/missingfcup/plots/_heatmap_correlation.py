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
            ignore_high_missingness=self.ignore_high_missingness,
            high_missingness_threshold=self.high_missingness_threshold,
            drop_constant=self.drop_constant_columns,
            order_by_missingness=self.order_by_missingness,
            order=self.order,
            max_columns=self.max_columns,
        )
        if len(cols) < 2:
            raise ValueError("Not enough columns with varying missingness to compute correlation.")
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = self.data.mask_missing[cols].corr()
        return corr.loc[cols, cols]

    def _colorbar_config(self) -> dict:
        return dict(
            title=(
                "Missingness correlation"
                "<br><span style='font-size:10px'>blue = missing together | red = mutually exclusive</span>"
                "<br><span style='font-size:10px'>NaN = insufficient overlap</span>"
            ),
            tickmode="array",
            tickvals=[-1, 0, 1],
            ticktext=["One missing, other present", "Independent", "Missing together"],
        )

    def _hover_template(self) -> str:
        return "<b>%{y}</b> vs <b>%{x}</b><br>Missingness correlation: %{z:.2f}<extra></extra>"
