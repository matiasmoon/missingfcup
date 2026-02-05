from matplotlib.pyplot import title
import pandas as pd
import numpy as np
from typing import Optional
from functools import cached_property
from typing import List, Optional, Literal, Dict

class MissingData:
    """Core object that owns all missingness-related logic."""

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame is empty")

        self.data = df

    # ------------------------------------------------------------------
    # Core representations
    # ------------------------------------------------------------------

    @cached_property
    def missing_matrix(self) -> pd.DataFrame:
        """Boolean matrix: True = missing, False = present."""
        return self.data.isna()

    @cached_property
    def presence_matrix(self) -> np.ndarray:
        """Binary matrix: 1 = present, 0 = missing."""
        return (~self.missing_matrix).to_numpy(np.uint8)

    @property
    def column_names(self) -> list[str]:
        """List of column names in the dataset."""
        return self.data.columns.tolist()

    # ------------------------------------------------------------------
    # Column / row summaries
    # ------------------------------------------------------------------

    @cached_property
    def missingness_per_column(self) -> pd.Series:
        """Fraction of missing values per column."""
        return self.missing_matrix.mean()

    @cached_property
    def missingness_per_row(self) -> pd.Series:
        """Fraction of missing values per row."""
        return self.missing_matrix.mean(axis=1)

    @cached_property
    def missing_count_per_column(self) -> pd.Series:
        """Number of missing values per column."""
        return self.missing_matrix.sum()

    @cached_property
    def missing_percentage_per_column(self) -> pd.Series:
        """Percentage of missing values per column."""
        return self.missingness_per_column * 100

    @cached_property
    def completeness_per_column(self) -> pd.Series:
        """Fraction of present (non-missing) values per column."""
        return 1 - self.missingness_per_column

    @cached_property
    def total_missingness(self) -> float:
        """Overall fraction of missing values in the dataset."""
        return self.missing_matrix.values.mean()

    @cached_property
    def has_missing_per_row(self) -> pd.Series:
        """Boolean indicator of whether each row contains any missing values."""
        return self.missing_matrix.any(axis=1)

    @cached_property
    def complete_rows(self) -> pd.Index:
        """Index of rows with no missing values."""
        return self.has_missing_per_row.loc[lambda s: ~s].index

    @cached_property
    def complete_columns(self) -> pd.Index:
        """Index of columns with no missing values."""
        return self.missing_matrix.any(axis=0).loc[lambda s: ~s].index

    # ------------------------------------------------------------------
    # Pattern analysis helpers
    # ------------------------------------------------------------------

    def missingness_correlation(self) -> pd.DataFrame:
        """Correlation matrix of missingness between columns."""
        return self.missing_matrix.corr()

    def missing_patterns_per_row(self) -> pd.Series:
        """Human-readable description of which columns are missing per row."""
        return self.missing_matrix.apply(
            lambda row: (
                "No values missing"
                if not row.any()
                else ", ".join(row.index[row])
            ),
            axis=1,
        )

    def missing_pattern_counts_per_row(
        self, max_patterns: Optional[int] = None
    ) -> pd.Series:
        """Count how many rows share the same missingness pattern."""
        counts = self.missing_patterns_per_row().value_counts()
        return counts if max_patterns is None else counts.head(max_patterns)

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def drop_columns_above_missingness(self, threshold: float) -> pd.DataFrame:
        """Drop columns whose missingness fraction exceeds a threshold."""
        cols = self.missingness_per_column <= threshold
        return self.data.loc[:, cols]

    def drop_rows_above_missingness(self, threshold: float) -> pd.DataFrame:
        """Drop rows whose missingness fraction exceeds a threshold."""
        rows = self.missingness_per_row <= threshold
        return self.data.loc[rows]

    # ------------------------------------------------------------------
    # Filtering & ordering helpers
    # ------------------------------------------------------------------
    def _filter_and_order(
        self,
        selected_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.9,
        completeness_mode: Optional[Literal["most", "least"]] = None,
        completeness_threshold: float = 0.0,
        max_columns_by_completeness: int = 0,
        max_columns: int = 50,
        order_by: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        """
        Filter, limit, and order the dataset before plotting.
        """
        df = self.data.copy()
        missing_fraction = df.isna().mean()
        completeness = 1 - missing_fraction

        # High missingness exclusion
        if ignore_high_missingness:
            keep = missing_fraction < high_missingness_threshold
            df = df.loc[:, keep]

        # Column selection
        if selected_columns:
            cols = [c for c in selected_columns if c in df.columns]
            df = df[cols]

        # Completeness-based filtering
        if completeness_mode:
            if completeness_mode == "most":
                df = df.loc[:, completeness >= completeness_threshold]
                if max_columns_by_completeness > 0:
                    idx = np.argsort(completeness)[-max_columns_by_completeness:]
                    df = df.iloc[:, np.sort(idx)]
            elif completeness_mode == "least":
                df = df.loc[:, completeness <= completeness_threshold]
                if max_columns_by_completeness > 0:
                    idx = np.argsort(completeness)[:max_columns_by_completeness]
                    df = df.iloc[:, np.sort(idx)]

        # Max columns limit
        if df.shape[1] > max_columns:
            df = df.iloc[:, :max_columns]

        # Ordering
        if order_by:
            for spec in reversed(order_by):
                col = spec["column"]
                ascending = spec.get("numeric_order", "asc") == "asc"
                if "ascending" in spec:
                    ascending = bool(spec["ascending"])

                if col == "__missing__":
                    missing_fraction = df.isna().mean()
                    ordered_cols = missing_fraction.sort_values(
                        ascending=ascending, kind="stable"
                    ).index
                    df = df.loc[:, ordered_cols]
                    continue

                if col == "__column__":
                    ordered_cols = sorted(df.columns, reverse=not ascending)
                    df = df.loc[:, ordered_cols]
                    continue

                if spec["type"] == "numeric":
                    df = df.sort_values(col, ascending=ascending, kind="stable")
                elif spec["type"] == "categorical":
                    cat = pd.Categorical(
                        df[col], categories=spec["category_order"], ordered=True
                    )
                    df = df.assign(**{col: cat}).sort_values(col, kind="stable")

        return df

    # ------------------------------------------------------------------
    # Plot functions
    # ------------------------------------------------------------------

    def _plot(self, cls, *args, **kwargs):
        """Instantiate a plot object"""
        return cls(self, *args, **kwargs)

    def barchart(
        self,
        selected_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.9,
        completeness_mode: Optional[Literal["most", "least"]] = None,
        completeness_threshold: float = 0.0,
        max_columns_by_completeness: int = 0,
        max_columns: int = 50,
        order_by: Optional[List[Dict]] = None,
        mode: Literal["count", "percentage"] = "count",
        orientation: Literal["vertical", "horizontal"] = "vertical",
        stacked: bool = False,
        missing_color: str = "#d62728",
        present_color: str = "#2ca02c",
        threshold: Optional[float] = None,
        threshold_color: str = "#ff7f0e",
        show_values: bool = True,
        title: Optional[str] = None,
    ):
        """Bar chart of missingness per column"""
        from ..plots.Barchart import BarChart
        return self._plot(
            BarChart,
            selected_columns=selected_columns,
            ignore_high_missingness=ignore_high_missingness,
            high_missingness_threshold=high_missingness_threshold,
            completeness_mode=completeness_mode,
            completeness_threshold=completeness_threshold,
            max_columns_by_completeness=max_columns_by_completeness,
            max_columns=max_columns,
            order_by=order_by,
            mode=mode,
            orientation=orientation,
            stacked=stacked,
            missing_color=missing_color,
            present_color=present_color,
            threshold=threshold,
            threshold_color=threshold_color,
            show_values=show_values,
            title=title,
        )

    def heatmap(self, **kwargs):
        """Heatmap of missingness"""
        from ..plots.Heatmap import Heatmap
        return self._plot(Heatmap, **kwargs)

    def scatterplot(self, x: str, y: str, *, point_size: int = 8, axis_padding: float = 0.05, title: Optional[str] = None, **kwargs):
        """Scatter plot of two columns, highlighting missingness."""
        from ..plots.Scatterplot import ScatterPlot
        return self._plot(ScatterPlot,x=x, y=y, point_size=point_size, axis_padding=axis_padding, title=title, **kwargs)

    def pattern_barchart(self, *, title: Optional[str] = None, max_patterns: Optional[int] = None):
        """Bar chart summarizing common missingness patterns"""
        from ..plots.PatternBarChart import PatternBarChart
        return self._plot(PatternBarChart, title=title, max_patterns=max_patterns)
    
    def missingness_correlation_heatmap(
        self,
        selected_columns: Optional[List[str]] = None,
        title: Optional[str] = None,
        **kwargs,
        ):
        """Heatmap of correlation between column missingness patterns"""
        from ..plots.CorrelationHeatmap import CorrelationHeatmap
        return self._plot(CorrelationHeatmap, selected_columns=selected_columns, title=title, **kwargs,)
    
    def column_missing_rate_heatmap(
        self,
        selected_columns: Optional[List[str]] = None,
        title: Optional[str] = None,
        **kwargs,
    ):
        """Heatmap showing missing rate per column."""
        from ..plots.ColumnMissingRateHeatmap import ColumnMissingRateHeatmap

        return self._plot(
            ColumnMissingRateHeatmap,
            selected_columns=selected_columns,
            title=title,
            **kwargs,
        )