import math
import pandas as pd
import numpy as np
from functools import cached_property
from typing import Optional, List, Dict, Literal, TYPE_CHECKING

from pandas.api.types import is_numeric_dtype
if TYPE_CHECKING:
    from missingfcup.plots.MissingCountBarChart import MissingCountBarChart

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
    def present_mask(self) -> pd.DataFrame:
        """
        Boolean mask of present (non-missing) values.

        True indicates a present value.
        False indicates a missing value.
        """
        return ~self.missing_mask

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

        Each tuple lists the columns missing in that specific row.
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

    @cached_property
    def present_missing_correlation(self) -> pd.DataFrame:
        """
        Correlation between present indicators and missing indicators.

        Returns a matrix where rows represent "present" for a column and
        columns represent "missing" for a column. Values are Pearson
        correlations in [-1, 1], with NaN for constant columns.
        """
        present = self.present_mask.astype(float)
        missing = self.missing_mask.astype(float)

        corr = pd.DataFrame(
            index=present.columns,
            columns=missing.columns,
            dtype=float,
        )

        for col_present in present.columns:
            x = present[col_present]
            x_std = x.std()
            for col_missing in missing.columns:
                y = missing[col_missing]
                y_std = y.std()
                if x_std == 0 or y_std == 0:
                    corr.loc[col_present, col_missing] = np.nan
                else:
                    corr.loc[col_present, col_missing] = x.corr(y)

        return corr

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

    def heatmap(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.95,
        max_columns: int = 50,
        order_by: Optional[List[Dict]] = None,
        show_colorscale: bool = False,
        group_by_mode: Literal["binary", "missing"] = "binary",
        xgap: int = 1,
        ygap: int = 0,
        max_label_length: int = 48,
        order_by_border_color: str = "#1f77b4",
        order_by_border_width: int = 5,
        **kwargs,
    ) -> "Heatmap":
        """Create an interactive missingness heatmap."""
        from ..plots.Heatmap import Heatmap

        return Heatmap(
            data=self,
            selected_columns=selected_columns,
            ignore_high_missingness=ignore_high_missingness,
            high_missingness_threshold=high_missingness_threshold,
            max_columns=max_columns,
            order_by=order_by,
            show_colorscale=show_colorscale,
            group_by_mode=group_by_mode,
            xgap=xgap,
            ygap=ygap,
            max_label_length=max_label_length,
            order_by_border_color=order_by_border_color,
            order_by_border_width=order_by_border_width,
            **kwargs,
        )

    def scatterplot(
        self,
        x: str,
        y: str,
        *,
        point_size: int = 8,
        axis_padding: float = 0.05,
        missingness_color_column: Optional[str] = None,
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
            missingness_color_column=missingness_color_column,
            **kwargs,
        )

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

    def upset_plot(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        max_sets: int = 8,
        max_intersections: int = 20,
        min_intersection_size: int = 1,
        sort_order: Literal["desc", "asc"] = "desc",
        show_values: bool = True,
        matrix_dot_size: int = 12,
        matrix_line_width: int = 3,
        excluded_dot_color: str = "#e0e0e0",
        highlight_columns: Optional[List[str]] = None,
        highlight_color: Optional[str] = None,
        **kwargs,
    ) -> "UpSetPlot":
        """Create an UpSet-style plot of missingness intersections."""
        from ..plots.UpSetPlot import UpSetPlot

        return UpSetPlot(
            data=self,
            selected_columns=selected_columns,
            max_sets=max_sets,
            max_intersections=max_intersections,
            min_intersection_size=min_intersection_size,
            sort_order=sort_order,
            show_values=show_values,
            matrix_dot_size=matrix_dot_size,
            matrix_line_width=matrix_line_width,
            excluded_dot_color=excluded_dot_color,
            highlight_columns=highlight_columns,
            highlight_color=highlight_color,
            **kwargs,
        )

    def parallel_coordinates(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        missingness_color_column: Optional[str] = None,
        normalize: bool = True,
        missingness_only: bool = False,
        impute_below_range_frac: float = 0.1,
        show_colorbar: bool = False,
        line_opacity: float = 0.5,
        **kwargs,
    ) -> "ParallelCoordinates":
        """Create a parallel coordinates plot with missing values imputed.

        If normalize=False, missing values are imputed below the observed range.
        If normalize=True, missing values are placed at the bottom of the
        normalized range (0.0).
        """
        from ..plots.ParallelCoordinates import ParallelCoordinates

        return ParallelCoordinates(
            data=self,
            selected_columns=selected_columns,
            missingness_color_column=missingness_color_column,
            normalize=normalize,
            missingness_only=missingness_only,
            impute_below_range_frac=impute_below_range_frac,
            show_colorbar=show_colorbar,
            line_opacity=line_opacity,
            **kwargs,
        )

    def missingness_correlation_heatmap(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        colorscale: str = "RdBu",
        show_values: bool = True,
        max_columns: int = 30,
        drop_constant_columns: bool = True,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 1,
        show_colorbar: bool = True,
        show_upper_triangle: bool = True,
        nan_color: str = "#c7c7c7",
        **kwargs,
    ) -> "CorrelationHeatmap":
        """Create a heatmap of missingness correlations between columns."""
        from ..plots.CorrelationHeatmap import CorrelationHeatmap

        return CorrelationHeatmap(
            data=self,
            selected_columns=selected_columns,
            colorscale=colorscale,
            show_values=show_values,
            max_columns=max_columns,
            drop_constant_columns=drop_constant_columns,
            order_by_missingness=order_by_missingness,
            order=order,
            value_round=value_round,
            show_colorbar=show_colorbar,
            show_upper_triangle=show_upper_triangle,
            nan_color=nan_color,
            **kwargs,
        )

    def all_correlation_heatmap(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        colorscale: str = "RdBu",
        show_values: bool = True,
        max_columns: int = 30,
        drop_constant_columns: bool = False,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 1,
        show_colorbar: bool = True,
        show_upper_triangle: bool = True,
        nan_color: str = "#c7c7c7",
        **kwargs,
    ) -> "AllCorrelationHeatmap":
        """Create a heatmap of present-vs-missing correlations for all columns."""
        from ..plots.AllCorrelationHeatmap import AllCorrelationHeatmap

        return AllCorrelationHeatmap(
            data=self,
            selected_columns=selected_columns,
            colorscale=colorscale,
            show_values=show_values,
            max_columns=max_columns,
            drop_constant_columns=drop_constant_columns,
            order_by_missingness=order_by_missingness,
            order=order,
            value_round=value_round,
            show_colorbar=show_colorbar,
            show_upper_triangle=show_upper_triangle,
            nan_color=nan_color,
            **kwargs,
        )
    
    def column_missing_rate_heatmap(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        scale: Literal["fraction", "percentage"] = "fraction",
        colorscale: str = "Reds",
        show_values: bool = True,
        max_columns: int = 30,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 2,
        show_colorbar: bool = True,
        max_label_length: int = 48,
        max_labels_with_values: int = 20,
        **kwargs,
    ) -> "ColumnMissingRateHeatmap":
        """Create a heatmap of missing rates per column."""
        from ..plots.ColumnMissingRateHeatmap import ColumnMissingRateHeatmap

        return ColumnMissingRateHeatmap(
            data=self,
            selected_columns=selected_columns,
            scale=scale,
            colorscale=colorscale,
            show_values=show_values,
            max_columns=max_columns,
            order_by_missingness=order_by_missingness,
            order=order,
            value_round=value_round,
            show_colorbar=show_colorbar,
            max_label_length=max_label_length,
            max_labels_with_values=max_labels_with_values,
            **kwargs,
        )

    def dendrogram(
        self,
        *,
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
    ) -> "Dendrogram":
        """Create a dendrogram of missingness correlation between columns."""
        from ..plots.Dendrogram import Dendrogram

        return Dendrogram(
            data=self,
            selected_columns=selected_columns,
            max_columns=max_columns,
            drop_constant_columns=drop_constant_columns,
            linkage_method=linkage_method,
            use_abs_correlation=use_abs_correlation,
            line_width=line_width,
            line_color=line_color,
            **kwargs,
        )

    # MARK: - Statistical Tests

    def littles_mcar_test(
        self,
        *,
        columns: Optional[List[str]] = None,
        numeric_only: bool = True,
        use_pairwise_cov: bool = True,
        regularize: float = 1e-6,
        max_iter: int = 200,
        tol: float = 1e-10,
    ) -> pd.Series:
        """
        Perform Little's MCAR (Missing Completely At Random) test.

        Parameters
        ----------
        columns : list[str], optional
            Columns to include in the test. If None, uses all columns.
        numeric_only : bool, default True
            If True, keeps only numeric columns (recommended).
        use_pairwise_cov : bool, default True
            If True, computes covariance pairwise using all available rows per pair.
            If False, uses only complete cases across selected columns.
        regularize : float, default 1e-6
            Diagonal ridge added to covariance submatrices for numerical stability.
        max_iter : int, default 200
            Maximum iterations for incomplete gamma approximation.
        tol : float, default 1e-10
            Convergence tolerance for incomplete gamma approximation.

        Returns
        -------
        pandas.Series
            Contains chi2 statistic, degrees of freedom, p-value,
            number of patterns, and columns used.

        Notes
        -----
        - The test assumes approximately multivariate normal data.
        - Non-numeric columns are ignored when numeric_only=True.
        """
        df = self.data if columns is None else self.data.loc[:, columns]

        if numeric_only:
            df = df.loc[:, [c for c in df.columns if is_numeric_dtype(df[c])]]

        if df.empty or df.shape[1] == 0:
            raise ValueError("No usable columns for Little's MCAR test.")

        x = df.to_numpy(dtype=float)
        mask = ~np.isnan(x)

        # Drop columns with all missing values
        col_has_obs = mask.any(axis=0)
        x = x[:, col_has_obs]
        mask = mask[:, col_has_obs]
        used_columns = df.columns[col_has_obs].tolist()

        if x.shape[1] == 0:
            raise ValueError("All selected columns are fully missing.")

        # Overall mean (column-wise)
        mu = np.nanmean(x, axis=0)

        # Covariance estimate
        if use_pairwise_cov:
            p = x.shape[1]
            cov = np.full((p, p), np.nan, dtype=float)
            for i in range(p):
                xi = x[:, i]
                for j in range(i, p):
                    xj = x[:, j]
                    valid = ~np.isnan(xi) & ~np.isnan(xj)
                    if valid.sum() <= 1:
                        cov_ij = np.nan
                    else:
                        cov_ij = np.cov(xi[valid], xj[valid], ddof=1)[0, 1]
                    cov[i, j] = cov_ij
                    cov[j, i] = cov_ij
        else:
            complete = mask.all(axis=1)
            if complete.sum() <= 1:
                raise ValueError("Not enough complete rows to estimate covariance.")
            cov = np.cov(x[complete], rowvar=False, ddof=1)

        # Replace NaNs in covariance with zeros (fallback for sparse pairs)
        cov = np.nan_to_num(cov, nan=0.0)

        # Build pattern groups
        pattern_keys = {}
        for idx, row_mask in enumerate(mask):
            key = tuple(row_mask.tolist())
            pattern_keys.setdefault(key, []).append(idx)

        chi2 = 0.0
        df_total = 0
        patterns_used = 0

        for key, row_idx in pattern_keys.items():
            obs_idx = [i for i, observed in enumerate(key) if observed]
            if len(obs_idx) == 0:
                continue
            rows = np.array(row_idx, dtype=int)
            xg = x[np.ix_(rows, obs_idx)]
            if xg.size == 0:
                continue
            mean_g = xg.mean(axis=0)
            mean_all = mu[obs_idx]

            sg = cov[np.ix_(obs_idx, obs_idx)]
            if regularize > 0:
                sg = sg + np.eye(len(obs_idx)) * regularize

            try:
                inv_sg = np.linalg.pinv(sg)
            except np.linalg.LinAlgError:
                continue

            diff = mean_g - mean_all
            chi2 += len(rows) * float(diff.T @ inv_sg @ diff)
            df_total += len(obs_idx)
            patterns_used += 1

        p = x.shape[1]
        df_stat = df_total - p
        if df_stat <= 0:
            raise ValueError("Degrees of freedom <= 0; not enough information for the test.")

        p_value = self._chi2_sf(chi2, df_stat, max_iter=max_iter, tol=tol)

        return pd.Series(
            {
                "chi2": chi2,
                "df": df_stat,
                "p_value": p_value,
                "n_patterns": patterns_used,
                "n_rows": x.shape[0],
                "n_columns": x.shape[1],
                "columns_used": used_columns,
            }
        )

    @staticmethod
    def _chi2_sf(x: float, k: int, *, max_iter: int = 200, tol: float = 1e-10) -> float:
        """Survival function (1-CDF) for chi-square using regularized gamma."""
        if x < 0 or k <= 0:
            return float("nan")
        a = 0.5 * k
        z = 0.5 * x
        if z == 0:
            return 1.0
        return MissingData._gammaincc(a, z, max_iter=max_iter, tol=tol)

    @staticmethod
    def _gammaincc(a: float, x: float, *, max_iter: int = 200, tol: float = 1e-10) -> float:
        """Regularized upper incomplete gamma Q(a, x)."""
        if x < 0 or a <= 0:
            return float("nan")

        # Series for P(a, x) when x < a + 1
        if x < a + 1.0:
            ap = a
            summation = 1.0 / a
            delta = summation
            for _ in range(max_iter):
                ap += 1.0
                delta *= x / ap
                summation += delta
                if abs(delta) < abs(summation) * tol:
                    break
            log_term = -x + a * math.log(x) - math.lgamma(a)
            p = summation * math.exp(log_term)
            return max(0.0, 1.0 - p)

        # Continued fraction for Q(a, x) when x >= a + 1
        b = x + 1.0 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, max_iter + 1):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < tol:
                break
        log_term = -x + a * math.log(x) - math.lgamma(a)
        q = h * math.exp(log_term)
        return max(0.0, min(1.0, q))
