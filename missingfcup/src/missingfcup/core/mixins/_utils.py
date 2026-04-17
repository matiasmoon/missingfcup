from __future__ import annotations

import math
from functools import cached_property
from typing import Optional, List
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class _MissingDataUtilsMixin:
    """
    Correlation matrices, missingness pattern analysis, and statistical tests.

    Depends on mask_missing and mask_present from _MissingDataCoreMixin,
    and data from MissingData.__init__, declared here as type hints only.
    """

    mask_missing: pd.DataFrame
    mask_present: pd.DataFrame
    data: pd.DataFrame

    # ------------------------------------------------------------------
    # Correlation matrices
    # ------------------------------------------------------------------

    @cached_property
    def missing_corr(self) -> pd.DataFrame:
        """
        Pearson correlation matrix of column missingness masks (missing–missing).

        Measures whether columns tend to be missing at the same time.

        +1.0 = columns always miss together
         0.0 = missingness is independent
        -1.0 = when one is missing the other is always present
        """
        return self.mask_missing.corr()

    @cached_property
    def present_present_corr(self) -> pd.DataFrame:
        """
        Pearson correlation matrix of column presence masks (present–present).

        Measures whether columns tend to be observed at the same time.

        +1.0 = columns always observed together
         0.0 = presence is independent
        -1.0 = when one is observed the other is always missing
        """
        return self.mask_present.astype(float).corr()

    @cached_property
    def present_missing_corr(self) -> pd.DataFrame:
        """
        Correlation between present indicators (rows) and missing indicators (columns).

        cell[i, j] = Pearson correlation between "column i is present" and
        "column j is missing". Reveals whether observing a value in one column
        predicts missingness in another — key for MAR diagnosis.

        Values are in [-1, 1]; NaN indicates a constant column.
        """
        present = self.mask_present.astype(float)
        missing = self.mask_missing.astype(float)

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

    # ------------------------------------------------------------------
    # Pattern analysis
    # ------------------------------------------------------------------

    @cached_property
    def missing_pattern_in_rows(self) -> pd.Series:
        """
        For each row, a tuple of column names that are missing.

        Two rows share the same pattern if they are missing values in exactly
        the same set of columns. Rows with no missing values have an empty tuple.
        """
        return self.mask_missing.apply(
            lambda row: tuple(row.index[row]),
            axis=1,
        )

    @cached_property
    def missing_pattern_in_rows_unique(self) -> pd.Index:
        """Unique row-level missingness patterns observed in the dataset."""
        return self.missing_pattern_in_rows.unique()

    def missing_pattern_counts(self, max_patterns: Optional[int] = None) -> pd.Series:
        """
        Frequency of each row-level missingness pattern, sorted descending.

        Parameters
        ----------
        max_patterns : int, optional
            If provided, limits output to the top N most frequent patterns.
        """
        counts = self.missing_pattern_in_rows.value_counts()
        return counts if max_patterns is None else counts.head(max_patterns)

    def perfectly_correlated_missing_columns(self) -> list[tuple[str, str]]:
        """
        Column pairs whose missingness patterns are perfectly correlated (r = 1.0).

        Identifies columns that are always missing in the exact same rows.
        """
        corr = self.missing_corr
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr.iloc[i, j] == 1:
                    pairs.append((cols[i], cols[j]))
        return pairs

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

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

        col_has_obs = mask.any(axis=0)
        x = x[:, col_has_obs]
        mask = mask[:, col_has_obs]
        used_columns = df.columns[col_has_obs].tolist()

        if x.shape[1] == 0:
            raise ValueError("All selected columns are fully missing.")

        mu = np.nanmean(x, axis=0)

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

        cov = np.nan_to_num(cov, nan=0.0)

        pattern_keys: dict = {}
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
        return _MissingDataUtilsMixin._gammaincc(a, z, max_iter=max_iter, tol=tol)

    @staticmethod
    def _gammaincc(a: float, x: float, *, max_iter: int = 200, tol: float = 1e-10) -> float:
        """Regularized upper incomplete gamma Q(a, x)."""
        if x < 0 or a <= 0:
            return float("nan")

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
