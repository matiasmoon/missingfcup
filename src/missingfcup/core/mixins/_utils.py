from __future__ import annotations

import math
from functools import cached_property
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class _MissingDataUtilsMixin:
    """
    Correlation matrices, missingness pattern analysis, and statistical tests.

    Depends on mask_missing from _MissingDataCoreMixin, and data from
    MissingData.__init__, declared here as type hints only.
    """

    mask_missing: pd.DataFrame
    data: pd.DataFrame

    # Correlation matrices

    @cached_property
    def missing_corr(self) -> pd.DataFrame:
        """
        Pearson correlation matrix of column missingness masks (missing/missing).

        Measures whether columns tend to be missing at the same time.

        +1.0 = columns always miss together
         0.0 = missingness is independent
        -1.0 = when one is missing the other is always present

        Correlating the *presence* masks instead returns this same matrix, since
        corr(1 - x, 1 - y) == corr(x, y), so presence needs no separate metric.
        """
        return self.mask_missing.corr()

    @cached_property
    def value_missing_corr(self) -> pd.DataFrame:
        """
        Point-biserial correlation between column values and missingness indicators.

        cell[i, j] = Pearson correlation between the observed values of column i
        and the binary missingness indicator of column j (1=missing, 0=present).
        Only rows where column i is observed are used (pairwise complete cases).

        Positive value means higher values of i associate with j being missing.
        Negative value means lower values of i associate with j being missing.
        NaN means column i is non-numeric or constant, or j has no variance in missingness.
        """
        # One row of the matrix at a time, rather than one cell. The rows kept for a
        # given value column depend only on that column's own NaNs, so for a fixed
        # row the whole set of missingness columns shares one valid-row mask and the
        # correlations against all of them are a single centred dot product. The
        # per-pair loop this replaces was quadratic in Python and measured ~6x slower
        # on a 30000x24 frame for bit-identical output.
        missing = self.mask_missing.to_numpy(dtype=float)

        corr = pd.DataFrame(
            index=self.data.columns,
            columns=self.mask_missing.columns,
            dtype=float,
        )

        for col_val in self.data.columns:
            x = pd.to_numeric(self.data[col_val], errors="coerce").to_numpy(dtype=float)
            valid = ~np.isnan(x)
            # Fewer than two observed values leaves nothing to correlate, and a
            # constant column has zero variance; both stay NaN, as before.
            if valid.sum() < 2:
                continue

            x_centred = x[valid] - x[valid].mean()
            y_centred = missing[valid] - missing[valid].mean(axis=0)

            with np.errstate(invalid="ignore", divide="ignore"):
                denominator = np.sqrt((x_centred @ x_centred) * (y_centred * y_centred).sum(axis=0))
                corr.loc[col_val] = np.where(
                    denominator > 0, (x_centred @ y_centred) / denominator, np.nan
                )

        return corr

    @cached_property
    def value_missing_dependence(self) -> pd.DataFrame:
        """
        Unsigned association between column values and missingness indicators.

        cell[i, j] = how far column j's missingness departs from independence of
        column i's observed values, on a 0-1 scale where 0 is independence and 1 is
        a perfect relationship. Only rows where column i is observed are used.

        The statistic is chosen by column i's dtype, because the right measure of
        "these are related" differs by what kind of variable it is:

        * numeric i  -> the two-sample Kolmogorov-Smirnov statistic, the largest gap
          between the distribution of i where j is present and where j is missing.
        * non-numeric i -> Cramer's V between i's categories and j's missingness.

        Both are unsigned distances from independence on the same 0-1 scale, so a
        single grid may mix them. This is the companion to :attr:`value_missing_corr`
        rather than a replacement: that one is signed and says *which way* a
        relationship runs, at the cost of only seeing relationships that have a
        direction. This one sees any departure from independence, including
        missingness concentrated at both tails of a column at once, where the two
        groups share a mean and a signed correlation reports nothing.

        NaN means column i has too few observed values, or column j's missingness
        never varies.

        Note that a nominal variable stored as integer codes is read as numeric, so
        it takes the KS branch. KS still detects the dependence -- it cannot report
        zero where one exists -- however its magnitude depends on the arbitrary
        order of the codes, where Cramer's V does not. Store such columns as
        ``category`` or string to get the label-invariant reading.
        """
        missing = self.mask_missing.to_numpy(dtype=bool)

        out = pd.DataFrame(
            index=self.data.columns,
            columns=self.mask_missing.columns,
            dtype=float,
        )

        for col_val in self.data.columns:
            series = self.data[col_val]
            if is_numeric_dtype(series):
                out.loc[col_val] = self._ks_row(series.to_numpy(dtype=float), missing)
            else:
                out.loc[col_val] = self._cramers_v_row(series, missing)

        return out

    @staticmethod
    def _ks_row(x: np.ndarray, missing: np.ndarray) -> np.ndarray:
        """KS statistic of ``x`` against every missingness column at once.

        Sorting ``x`` once and walking the two empirical distribution functions
        together turns a per-pair scipy call into one pass over the sorted mask,
        which measured ~90x faster on a 30000x24 frame for the same numbers.
        """
        valid = ~np.isnan(x)
        result = np.full(missing.shape[1], np.nan)
        if valid.sum() < 2:
            return result

        order = np.argsort(x[valid], kind="mergesort")
        x_sorted = x[valid][order]
        mask_sorted = missing[valid][order]

        n_missing = mask_sorted.sum(axis=0)
        n_present = (~mask_sorted).sum(axis=0)
        usable = (n_missing > 0) & (n_present > 0)
        if not usable.any():
            return result

        # The two distribution functions are only comparable at the end of a run of
        # equal values: inside a tie the step is not yet complete, and reading it
        # there would overstate the gap.
        run_end = np.empty(len(x_sorted), dtype=bool)
        run_end[:-1] = x_sorted[:-1] != x_sorted[1:]
        run_end[-1] = True

        with np.errstate(invalid="ignore", divide="ignore"):
            cdf_missing = np.cumsum(mask_sorted, axis=0) / n_missing
            cdf_present = np.cumsum(~mask_sorted, axis=0) / n_present
            gap = np.abs(cdf_present - cdf_missing)[run_end]
        result[usable] = gap[:, usable].max(axis=0)
        return result

    @staticmethod
    def _cramers_v_row(series: pd.Series, missing: np.ndarray) -> np.ndarray:
        """Cramer's V of a categorical column against every missingness column.

        The missingness side always has two levels, so ``min(rows, cols) - 1`` is 1
        and V reduces to ``sqrt(chi2 / n)``. Unlike a correlation on category codes,
        this does not depend on what order the categories were numbered in.
        """
        codes, _ = pd.factorize(series, use_na_sentinel=True)
        valid = codes >= 0
        result = np.full(missing.shape[1], np.nan)
        n_valid = int(valid.sum())
        if n_valid < 2:
            return result

        codes = codes[valid]
        n_levels = int(codes.max()) + 1
        if n_levels < 2:
            # One category carries no information about anything.
            return np.where(missing[valid].any(axis=0), 0.0, np.nan)

        row_totals = np.bincount(codes, minlength=n_levels).astype(float)
        for j in range(missing.shape[1]):
            column = missing[valid][:, j]
            n_missing = int(column.sum())
            if n_missing == 0 or n_missing == n_valid:
                continue
            observed_missing = np.bincount(codes[column], minlength=n_levels).astype(float)
            observed = np.column_stack([row_totals - observed_missing, observed_missing])
            expected = np.outer(row_totals, np.array([n_valid - n_missing, n_missing])) / n_valid
            with np.errstate(invalid="ignore", divide="ignore"):
                terms = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
            result[j] = math.sqrt(terms.sum() / n_valid)
        return result

    # Pattern analysis

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

    # Statistical tests

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
        * The test assumes approximately multivariate normal data.
        * Non-numeric columns are ignored when numeric_only=True.
        """
        df = self.data if columns is None else self.data.loc[:, columns]

        if numeric_only:
            df = df.loc[:, [c for c in df.columns if is_numeric_dtype(df[c])]]

        if df.empty or df.shape[1] == 0:
            reason = "none of them are numeric" if numeric_only else "the selection is empty"
            raise ValueError(
                f"No usable columns for Little's MCAR test: {reason}. The test "
                f"compares means across missingness patterns, so it needs numeric "
                f"columns. Pass numeric_only=False to include the rest."
            )

        x = df.to_numpy(dtype=float)
        mask = ~np.isnan(x)

        col_has_obs = mask.any(axis=0)
        x = x[:, col_has_obs]
        mask = mask[:, col_has_obs]
        used_columns = df.columns[col_has_obs].tolist()

        if x.shape[1] == 0:
            raise ValueError(
                f"All {df.shape[1]} selected columns are fully missing, so there are "
                f"no observed values to compare. Little's test needs columns that are "
                f"at least partly observed."
            )

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
                raise ValueError(
                    f"Only {int(complete.sum())} rows have no missing values, which is "
                    f"too few to estimate a covariance matrix. Pass "
                    f"use_pairwise_cov=True to estimate it pairwise instead."
                )
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
            raise ValueError(
                f"Not enough information for Little's test: {patterns_used} missingness "
                f"pattern(s) across {p} columns give {df_stat} degrees of freedom. The "
                f"test needs more distinct patterns than columns."
            )

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

    def mann_whitney_test(
        self,
        x: str,
        by: str,
        *,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        alpha: float = 0.05,
        use_continuity: bool = True,
    ) -> pd.Series:
        """
        Mann-Whitney U test comparing the distribution of ``x`` between rows
        where ``by`` is present and rows where ``by`` is missing.

        This is the statistical counterpart to :meth:`boxplot` and :meth:`density`:
        it answers "does the value of ``x`` differ depending on whether ``by`` is
        observed?" with a p-value instead of a purely visual judgement.

        It is a non-parametric, two-sample rank test: it makes no normality
        assumption and only compares the stochastic ordering of the two groups,
        which makes it broadly applicable to the skewed, non-Gaussian variables
        common in real datasets.

        Diagnostic reading:

        * significant (p < alpha) -> the distribution of ``x`` depends on the
          missingness of ``by``; consistent with MAR (or MNAR if ``x`` == ``by``'s
          own latent value proxy). Missingness is not completely at random.
        * not significant           -> no detectable dependence; consistent with
          MCAR with respect to ``x`` (absence of evidence, not proof).

        Parameters
        ----------
        x : str
            Numeric column whose distribution is compared across the two groups.
        by : str
            Column whose missingness defines the two groups (present vs. missing).
        alternative : {"two-sided", "less", "greater"}, default "two-sided"
            Passed to ``scipy.stats.mannwhitneyu``. "less"/"greater" test whether
            the present-group distribution is stochastically smaller/larger.
        alpha : float, default 0.05
            Significance level used to set the boolean ``significant`` field.
        use_continuity : bool, default True
            Apply the continuity correction (relevant for the normal approximation).

        Returns
        -------
        pandas.Series
            U statistic, p-value, alpha, significance flag, per-group sample
            sizes and medians, and the tested column names.

        Notes
        -----
        Only rows where ``x`` is itself observed are used, evaluated separately
        within each ``by`` group (pairwise-complete). Requires at least one
        observed value of ``x`` in each group.
        """
        # Imported here, not at module scope: this mixin loads on `import
        # missingfcup`, and scipy.stats is slow to import for a rarely called test.
        from scipy.stats import mannwhitneyu

        for name, role in ((x, "x"), (by, "by")):
            if name not in self.data.columns:
                raise ValueError(
                    f"{role}={name!r} is not a column in the DataFrame. "
                    f"It holds {list(self.data.columns)}."
                )

        values = pd.to_numeric(self.data[x], errors="coerce")
        if values.notna().sum() == 0:
            raise ValueError(f"Column {x!r} has no usable numeric values.")

        by_missing = self.mask_missing[by]
        present_group = values[~by_missing].dropna()
        missing_group = values[by_missing].dropna()

        if len(present_group) == 0 or len(missing_group) == 0:
            raise ValueError(
                f"Not enough observed values of {x!r} in both groups of {by!r} "
                f"(present={len(present_group)}, missing={len(missing_group)})."
            )

        u_stat, p_value = mannwhitneyu(
            present_group,
            missing_group,
            alternative=alternative,
            use_continuity=use_continuity,
        )

        return pd.Series(
            {
                "U": float(u_stat),
                "p_value": float(p_value),
                "alpha": alpha,
                "significant": bool(p_value < alpha),
                "alternative": alternative,
                "n_present": int(len(present_group)),
                "n_missing": int(len(missing_group)),
                "median_present": float(present_group.median()),
                "median_missing": float(missing_group.median()),
                "value_column": x,
                "missingness_column": by,
            }
        )

    def ks_test(
        self,
        x: str,
        by: str,
        *,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        alpha: float = 0.05,
    ) -> pd.Series:
        """
        Two-sample Kolmogorov-Smirnov test comparing the distribution of ``x``
        between rows where ``by`` is present and rows where ``by`` is missing.

        This asks the same question as :meth:`mann_whitney_test` but measures a
        different thing, and the difference decides which one finds a given
        relationship. Mann-Whitney compares stochastic ordering: it detects that
        one group sits *above* the other. KS compares the two empirical
        distribution functions at every point and reports their largest gap, so it
        detects any difference in shape, including ones with no direction.

        That matters because missingness is often concentrated at *both* ends of a
        column. A sensor that clips at its limits, or a survey question skipped by
        respondents at either extreme of a scale, produces two groups with the same
        centre and very different spread. Mann-Whitney and the point-biserial cells
        of ``heatmap(kind="direction")`` both read that as no relationship; KS does
        not. Reach for it whenever a location-based test comes back flat on a
        column you have other reasons to suspect.

        Diagnostic reading:

        * significant (p < alpha) -> the distribution of ``x`` differs between the
          two groups; missingness in ``by`` is not independent of ``x``, which
          rules out MCAR with respect to ``x``.
        * not significant           -> no detectable difference in distribution;
          consistent with MCAR with respect to ``x`` (absence of evidence, not
          proof).

        A significant result here with a flat ``mann_whitney_test`` is the
        signature of a symmetric, two-tailed dependence.

        Parameters
        ----------
        x : str
            Numeric column whose distribution is compared across the two groups.
        by : str
            Column whose missingness defines the two groups (present vs. missing).
        alternative : {"two-sided", "less", "greater"}, default "two-sided"
            Passed to ``scipy.stats.ks_2samp``. The default is the one to use for
            diagnosis: the one-sided forms test a signed difference in the
            distribution functions and so give up the symmetric case that is the
            reason to prefer this test.
        alpha : float, default 0.05
            Significance level used to set the boolean ``significant`` field.

        Returns
        -------
        pandas.Series
            KS statistic (the largest gap between the two distribution functions,
            in [0, 1]), p-value, alpha, significance flag, per-group sample sizes
            and medians, and the tested column names.

        Notes
        -----
        Only rows where ``x`` is itself observed are used, evaluated separately
        within each ``by`` group (pairwise-complete). Requires at least one
        observed value of ``x`` in each group.

        The statistic is unsigned: it says how far apart the two groups are, never
        which one is higher. That is inherent to what it measures, since a
        two-tailed difference has no direction to report. Use
        ``heatmap(kind="direction")`` or ``mann_whitney_test`` when the direction is
        what you need.
        """
        # Imported here, not at module scope: this mixin loads on `import
        # missingfcup`, and scipy.stats is slow to import for a rarely called test.
        from scipy.stats import ks_2samp

        for name, role in ((x, "x"), (by, "by")):
            if name not in self.data.columns:
                raise ValueError(
                    f"{role}={name!r} is not a column in the DataFrame. "
                    f"It holds {list(self.data.columns)}."
                )

        values = pd.to_numeric(self.data[x], errors="coerce")
        if values.notna().sum() == 0:
            raise ValueError(f"Column {x!r} has no usable numeric values.")

        by_missing = self.mask_missing[by]
        present_group = values[~by_missing].dropna()
        missing_group = values[by_missing].dropna()

        if len(present_group) == 0 or len(missing_group) == 0:
            raise ValueError(
                f"Not enough observed values of {x!r} in both groups of {by!r} "
                f"(present={len(present_group)}, missing={len(missing_group)})."
            )

        statistic, p_value = ks_2samp(
            present_group,
            missing_group,
            alternative=alternative,
        )

        return pd.Series(
            {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "alpha": alpha,
                "significant": bool(p_value < alpha),
                "alternative": alternative,
                "n_present": int(len(present_group)),
                "n_missing": int(len(missing_group)),
                "median_present": float(present_group.median()),
                "median_missing": float(missing_group.median()),
                "value_column": x,
                "missingness_column": by,
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
