import pandas as pd
import numpy as np
from functools import cached_property
from typing import Optional, List, Dict, Literal, TYPE_CHECKING

from pandas.api.types import is_numeric_dtype
if TYPE_CHECKING:
    from ..plots.Barchart import MissingCountBarChart

class MissingData:
    """
    Analytical interface for inspecting and quantifying missing values in a pandas DataFrame.
    This class centralizes all missing-value related computations and exposes consistent, cached metrics at the column, row, and dataset levels.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame is empty")

        self.data = df

    # MARK: - Core Properties

    @cached_property
    def missing_mask(self) -> pd.DataFrame:
        """
        Boolean mask of missing values.

        True indicates a missing value.
        False indicates a present value.
        """
        return self.data.isna()

    @cached_property
    def observed_mask(self) -> np.ndarray:
        """
        Binary matrix representing observed values.

        0 indicates a missing value.
        1 indicates a present value.
        """
        return (~self.missing_mask).to_numpy(np.uint8)

    @property
    def columns(self) -> list[str]:
        return self.data.columns.tolist()

    @property
    def number_of_rows(self) -> int:
        return len(self.data)

    @property
    def number_of_columns(self) -> int:
        return self.data.shape[1]

    # MARK: - Column Metrics

    @cached_property
    def column_missing_rate(self) -> pd.Series:
        """
        Proportion of missing values per column.
        
        Returns values between 0 and 1.
        1.0 indicates all values in the column are missing.
        0.0 indicates no missing values in the column.
        """
        return self.missing_mask.mean()

    @cached_property
    def column_missing_count(self) -> pd.Series:
        """
        Number of missing values per column.
        """
        return self.missing_mask.sum()

    @cached_property
    def column_present_count(self) -> pd.Series:
        """
        Number of observed (non-missing) values per column.
        """
        return self.number_of_rows - self.column_missing_count

    @cached_property
    def column_missing_percent(self) -> pd.Series:
        """
        Percentage of missing values per column.
        """
        return self.column_missing_rate * 100

    @cached_property
    def column_completeness(self) -> pd.Series:
        """
        Proportion of observed (non-missing) values per column.
        """
        return 1 - self.column_missing_rate

    @cached_property
    def columns_complete(self) -> pd.Index:
        """
        Columns with no missing values.
        """
        return self.column_missing_count.loc[lambda s: s == 0].index

    # MARK: - Row Metrics

    @cached_property
    def row_missing_rate(self) -> pd.Series:
        """
        Proportion of missing values per row.
        
        Returns values between 0 and 1.
        1.0 indicates all values in the row are missing.
        0.0 indicates no missing values in the row.
        """
        return self.missing_mask.mean(axis=1)

    @cached_property
    def row_missing_count(self) -> pd.Series:
        """
        Number of missing values per row.
        """
        return self.missing_mask.sum(axis=1)

    @cached_property
    def row_missing_percent(self) -> pd.Series:
        """
        Percentage of missing values per row.
        """
        return self.row_missing_rate * 100

    @cached_property
    def row_completeness(self) -> pd.Series:
        """
        Proportion of observed (non-missing) values per row.
        """
        return 1 - self.row_missing_rate

    @cached_property
    def rows_complete(self) -> pd.Index:
        """
        Index labels of rows containing no missing values.
        """
        return self.row_missing_count.loc[lambda s: s == 0].index

    @cached_property
    def rows_with_missing(self) -> pd.Index:
        """
        Rows containing at least one missing value.
        """
        return self.row_missing_count.loc[lambda s: s > 0].index

    # MARK: - Dataset-Level Metrics

    @cached_property
    def total_missing_rate(self) -> float:
        """
        Overall fraction of missing values in the dataset.
        Represents the density of missingness across the entire matrix.
        """
        return self.missing_mask.values.mean()

    @cached_property
    def total_missing_count(self) -> int:
        """
        Total number of missing values in the dataset.
        """
        return int(self.missing_mask.values.sum())

    # MARK: - Pattern Analysis
    @cached_property
    def missing_pattern_in_rows(self) -> pd.Series:
        """
        For each row, returns a tuple of column names that are missing.

        Two rows share the same missing pattern if they are missing values in the exact same set of columns.

        Identifies the columns where there is missing values for every row.
        Rows with no missing values will have an empty tuple.
        """
        return self.missing_mask.apply(
            lambda row: tuple(row.index[row]),
            axis=1,
        )

    @cached_property
    def missing_pattern_in_rows_unique(self) -> pd.Index:
        """
        Returns the unique row-level missingness patterns observed in the dataset.
        """
        return self.missing_pattern_in_rows.unique()

    def missing_pattern_counts(self, max_patterns: Optional[int] = None) -> pd.Series:
        """
        Return counts of row-level missingness patterns.

        Parameters
        ----------
        max_patterns : int, optional
            Limit the output to the top `max_patterns` most frequent patterns if provided.

        Returns
        -------
        pandas.Series
            Pattern frequencies sorted in descending order.

        Examples
        --------
        - With `max_patterns=3`, only the three most common patterns are returned.
        """
        counts = self.missing_pattern_in_rows.value_counts()
        return counts if max_patterns is None else counts.head(max_patterns)

    def missing_correlation(self) -> pd.DataFrame:
        """
        The Pearson correlation matrix of column nullity masks.

        Returns
        -------
        pandas.DataFrame
            Correlation coefficients where values indicates:
            1.0 columns always miss together
            0 independence
            - 1 when one is present while the other is missing

        Examples
        --------
        - Columns that always go missing together will have correlation values of 1.0.
        """
        return self.missing_mask.corr()

    def perfectly_correlated_missing_columns(self) -> list[tuple[str, str]]:
        """
        Return column pairs whose missingness patterns are perfectly correlated.

        Returns
        -------
        list[tuple[str, str]]
            Tuples of column names with a missingness correlation coefficient of 1.0.

        Examples
        --------
        - Identifies `(col_a, col_b)` when both columns are missing exactly the same cells.
        """
        corr = self.missing_correlation()
        pairs = []
        cols = corr.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr.iloc[i, j] == 1:
                    pairs.append((cols[i], cols[j]))

        return pairs

    # MARK: - Filtering Utilities

    def rows_above_missing_threshold(self, threshold: float) -> pd.Index:
        """
        Return rows whose missing rate exceeds a specified threshold.

        Parameters
        ----------
        threshold : float
            Proportion of allowed missing values in [0, 1]; rows with a missing rate strictly greater than this value are returned.

        Returns
        -------
        pandas.Index
            Index labels of rows exceeding the threshold.

        Examples
        --------
        - 0.2 → rows missing more than 20% of their values
        - 0.5 → rows missing more than half their values
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        return self.row_missing_rate.loc[lambda s: s > threshold].index

    def columns_above_missing_threshold(self, threshold: float) -> pd.Index:
        """
        Return columns whose missing rate exceeds a specified threshold.

        Parameters
        ----------
        threshold : float
            Proportion of allowed missing values in [0, 1]; columns with a missing rate strictly greater than this value are returned.

        Returns
        -------
        pandas.Index
            Index labels of columns exceeding the threshold.

        Examples
        --------
        - 0.2 → columns missing more than 20% of their values
        - 0.5 → columns missing more than half their values
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        return self.column_missing_rate.loc[lambda s: s > threshold].index

    # # MARK: - Missingness Mechanism Diagnostics

    # @staticmethod
    # def _cramers_v(table: pd.DataFrame) -> float:
    #     """
    #     Compute Cramer's V for a contingency table.
    #     """
    #     if table.empty:
    #         return float("nan")
    #     observed = table.to_numpy(dtype=float)
    #     n = observed.sum()
    #     if n == 0:
    #         return float("nan")
    #     row_sum = observed.sum(axis=1, keepdims=True)
    #     col_sum = observed.sum(axis=0, keepdims=True)
    #     expected = row_sum @ col_sum / n
    #     with np.errstate(divide="ignore", invalid="ignore"):
    #         chi2 = np.nansum((observed - expected) ** 2 / expected)
    #     r, c = observed.shape
    #     k = min(r - 1, c - 1)
    #     if k <= 0:
    #         return float("nan")
    #     return float(np.sqrt(chi2 / (n * k)))

    # def missingness_associations(
    #     self,
    #     target_column: str,
    #     *,
    #     max_categories: int = 20,
    #     min_non_null: int = 30,
    # ) -> pd.DataFrame:
    #     """
    #     Measure associations between a target column's missingness and other columns.

    #     This is a diagnostic heuristic: strong associations suggest MAR; weak associations
    #     may indicate MCAR, but MNAR cannot be confirmed from observed data alone.

    #     Parameters
    #     ----------
    #     target_column : str
    #         Column whose missingness will be used as the target indicator.
    #     max_categories : int, optional
    #         Maximum number of categories for categorical predictors.
    #     min_non_null : int, optional
    #         Minimum non-null observations required for a predictor to be considered.

    #     Returns
    #     -------
    #     pandas.DataFrame
    #         Predictor-level associations sorted by absolute strength.
    #     """
    #     if target_column not in self.data.columns:
    #         raise KeyError(f"Column '{target_column}' not found in data.")

    #     missing_indicator = self.data[target_column].isna().astype(np.int8)
    #     if missing_indicator.nunique() < 2:
    #         return pd.DataFrame(
    #             columns=["predictor", "predictor_type", "metric", "value", "n"]
    #         )

    #     results: list[dict[str, object]] = []
    #     for column in self.data.columns:
    #         if column == target_column:
    #             continue
    #         series = self.data[column]
    #         mask = series.notna()
    #         if int(mask.sum()) < min_non_null:
    #             continue

    #         if is_numeric_dtype(series):
    #             valid = mask
    #             x = series.loc[valid]
    #             y = missing_indicator.loc[valid]
    #             if y.nunique() < 2 or x.nunique(dropna=True) < 2:
    #                 continue
    #             value = x.corr(y)
    #             if pd.isna(value):
    #                 continue
    #             results.append(
    #                 {
    #                     "predictor": column,
    #                     "predictor_type": "numeric",
    #                     "metric": "pearson_r",
    #                     "value": float(value),
    #                     "n": int(valid.sum()),
    #                 }
    #             )
    #         else:
    #             nunique = series.nunique(dropna=True)
    #             if nunique == 0 or nunique > max_categories:
    #                 continue
    #             ct = pd.crosstab(missing_indicator, series, dropna=True)
    #             if ct.shape[0] < 2 or ct.shape[1] < 2:
    #                 continue
    #             value = self._cramers_v(ct)
    #             if pd.isna(value):
    #                 continue
    #             results.append(
    #                 {
    #                     "predictor": column,
    #                     "predictor_type": "categorical",
    #                     "metric": "cramers_v",
    #                     "value": float(value),
    #                     "n": int(ct.to_numpy().sum()),
    #                 }
    #             )

    #     associations = pd.DataFrame(results)
    #     if associations.empty:
    #         return associations
    #     associations = associations.assign(abs_value=associations["value"].abs())
    #     return associations.sort_values("abs_value", ascending=False).drop(
    #         columns="abs_value"
    #     )

    # def missingness_mechanism_summary(
    #     self,
    #     *,
    #     corr_threshold: float = 0.2,
    #     cramer_v_threshold: float = 0.2,
    #     max_categories: int = 20,
    #     min_non_null: int = 30,
    # ) -> pd.DataFrame:
    #     """
    #     Summarize evidence for MCAR/MAR for each column with missing values.

    #     Notes
    #     -----
    #     - Strong associations with observed variables suggest MAR.
    #     - Weak associations may indicate MCAR, but this is not conclusive.
    #     - MNAR cannot be verified from observed data alone.
    #     """
    #     summaries: list[dict[str, object]] = []
    #     for column, rate in self.column_missing_rate.items():
    #         if rate == 0:
    #             continue
    #         associations = self.missingness_associations(
    #             column,
    #             max_categories=max_categories,
    #             min_non_null=min_non_null,
    #         )
    #         if associations.empty:
    #             summaries.append(
    #                 {
    #                     "column": column,
    #                     "missing_rate": float(rate),
    #                     "strongest_association": None,
    #                     "strongest_value": None,
    #                     "conclusion": "Insufficient evidence (MCAR possible; MNAR not testable)",
    #                 }
    #             )
    #             continue

    #         top = associations.iloc[0]
    #         threshold = (
    #             corr_threshold
    #             if top["metric"] == "pearson_r"
    #             else cramer_v_threshold
    #         )
    #         conclusion = (
    #             "MAR likely (missingness associated with observed variables)"
    #             if abs(top["value"]) >= threshold
    #             else "Weak associations (MCAR possible)"
    #         )
    #         summaries.append(
    #             {
    #                 "column": column,
    #                 "missing_rate": float(rate),
    #                 "strongest_association": top["predictor"],
    #                 "strongest_value": float(top["value"]),
    #                 "conclusion": conclusion,
    #             }
    #         )

    #     summary = pd.DataFrame(summaries)
    #     if summary.empty:
    #         return summary
    #     return summary.sort_values("missing_rate", ascending=False)

    # MARK: - Plot functions

    def missing_count_barchart(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.9,
        completeness_mode: Optional[Literal["most", "least"]] = None,
        completeness_threshold: float = 0.0,
        max_columns_by_completeness: int = 0,
        max_columns: int = 50,
        order_by: Optional[List[Dict]] = None,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        show_values: bool = True,
        value: Literal["missing", "present"] = "missing",
        show_both: bool = False,
        **kwargs,
    ) -> "MissingCountBarChart":
        """Create a bar chart of missing counts per column."""
        from ..plots.MissingCountBarChart import MissingCountBarChart

        return MissingCountBarChart(
            data=self,
            selected_columns=selected_columns,
            ignore_high_missingness=ignore_high_missingness,
            high_missingness_threshold=high_missingness_threshold,
            completeness_mode=completeness_mode,
            completeness_threshold=completeness_threshold,
            max_columns_by_completeness=max_columns_by_completeness,
            max_columns=max_columns,
            order_by=order_by,
            orientation=orientation,
            show_values=show_values,
            value=value,
            show_both=show_both,
            **kwargs,
        )

    # TODO: Implement heatmap method
    # def heatmap(self, **kwargs):
    #    return

    def scatterplot(
        self,
        x: str,
        y: str,
        *,
        point_size: int = 8,
        axis_padding: float = 0.05,
        **kwargs,
    ) -> "ScatterPlot":
        """Create a scatter plot highlighting missingness in two columns."""
        from ..plots.Scatterplot import ScatterPlot

        return ScatterPlot(
            data=self,
            x=x,
            y=y,
            point_size=point_size,
            axis_padding=axis_padding,
            **kwargs,
        )

    # TODO: Implement pattern_barchart method
    def pattern_barchart(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        sort_order: Literal["desc", "asc"] = "desc",
        value: Literal["count", "percent"] = "count",
        show_values: bool = True,
        max_label_length: int = 48,
        missing_color: str = "#d62728",
        **kwargs,
    ) -> "PatternBarChart":
        """Create a bar chart of row-level missingness patterns."""
        from ..plots.PatternBarChart import PatternBarChart

        return PatternBarChart(
            data=self,
            selected_columns=selected_columns,
            sort_order=sort_order,
            value=value,
            show_values=show_values,
            max_label_length=max_label_length,
            missing_color=missing_color,
            **kwargs,
        )
    
    # TODO: Implement missingness_correlation_heatmap method
    # def missingness_correlation_heatmap(self, selected_columns: Optional[List[str]] = None, title: Optional[str] = None, **kwargs, ):
    #    return
    
    # TODO: Implement column_missing_rate_heatmap method
    # def column_missing_rate_heatmap(self, selected_columns: Optional[List[str]] = None, title: Optional[str] = None, **kwargs,):
    #    return
