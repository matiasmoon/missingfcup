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
    
    @property
    def row_missing_pattern(self) -> pd.Series:
        """
        For each row, returns a string describing which columns are missing.
        Example:
        - "No values missing"
        - "Ozone"
        - "Ozone, Solar.R"
        """

        def pattern(row: pd.Series) -> str:
            missing_cols = row.index[row].tolist()
            if not missing_cols:
                return "No values missing"
            return ", ".join(missing_cols)

        return self.missing_matrix.apply(pattern, axis=1)

    def row_missing_pattern_counts(self, max_patterns: Optional[int] = None,) -> pd.Series:
        """
        Counts number of rows sharing the same missingness pattern.
        """
        counts = self.row_missing_pattern.value_counts()

        if max_patterns is not None:
            counts = counts.iloc[:max_patterns]

        return counts

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
    
    def pattern_barchart(
        self,
        *,
        title: Optional[str] = None,
        max_patterns: Optional[int] = None,
    ):
        from ..plots.PatternBarChart import PatternBarChart
        return PatternBarChart(
            data=self,
            title=title,
            max_patterns=max_patterns,
        )