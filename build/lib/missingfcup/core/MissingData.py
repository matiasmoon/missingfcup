import pandas as pd
import numpy as np
from typing import Optional
from .ViewMetadata import ViewMetadata

class MissingData:
    """
    Core object that owns missingness logic.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame is empty")

        self.data = df
        self.missing_matrix = df.isna()

    @property
    def values(self) -> np.ndarray:
        """
        Binary matrix:
        1 = present
        0 = missing
        """
        return (~self.missing_matrix).to_numpy(dtype=np.uint8)

    @property
    def columns(self) -> list[str]:
        return list(self.data.columns)

    @property
    def column_missingness(self) -> pd.Series:
        """Fraction of missing values per column"""
        return self.missing_matrix.mean(axis=0)

    @property
    def row_missingness(self) -> pd.Series:
        """Fraction of missing values per row"""
        return self.missing_matrix.mean(axis=1)

    @property
    def missing_counts(self) -> pd.Series:
        """Number of missing values per column"""
        return self.missing_matrix.sum()

    @property
    def missing_percentages(self) -> pd.Series:
        """Percentage of missing values per column"""
        return self.missing_counts / len(self.data) * 100

    @property
    def completeness(self) -> pd.Series:
        """Fraction of present values per column"""
        return 1 - self.column_missingness

    def heatmap(
        self,
        metadata: Optional[ViewMetadata] = None,
        **kwargs,
    ):
        from ..plots.Heatmap import Heatmap
        return Heatmap(self, metadata=metadata, **kwargs)

    def barchart(
        self,
        metadata: Optional[ViewMetadata] = None,
        **kwargs,
    ):
        from ..plots.Barchart import BarChart
        return BarChart(self, metadata=metadata, **kwargs)

    def scatterplot(
        self,
        x: str,
        y: str,
        metadata: Optional[ViewMetadata] = None,
        **kwargs,
    ):
        from ..plots.Scatterplot import ScatterPlot
        return ScatterPlot(self, x=x, y=y, metadata=metadata, **kwargs)
