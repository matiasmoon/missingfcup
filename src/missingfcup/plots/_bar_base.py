from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from missingfcup.core.missing_data import MissingData
from missingfcup.plots._ordering import normalize_order_by
from missingfcup.plots._plot import _Plot


class _BarBase(_Plot):
    """Shared column picking for the per-column bar charts (count and rate).

    Both bars select, filter and order their columns the same way. Only what they draw
    at the end differs, so a subclass just implements ``_build_figure`` and may add its
    own options (``value``/``show_both`` for counts, ``scale`` for rate).
    """

    def __init__(
        self,
        data: MissingData,
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
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.ignore_high_missingness = ignore_high_missingness
        self.high_missingness_threshold = high_missingness_threshold
        self.completeness_mode = completeness_mode
        self.completeness_threshold = completeness_threshold
        self.max_columns_by_completeness = max_columns_by_completeness
        self.max_columns = max_columns
        self.order_by = order_by
        self.orientation = orientation
        self.show_values = show_values

    def _prepare_df(self) -> pd.DataFrame:
        df = self.data.data.copy()
        df = self._apply_missingness_filter(df)
        df = self._apply_column_selection(df)
        df = self._apply_completeness_filter(df)
        df = self._apply_max_columns_limit(df)
        df = self._apply_ordering(df)
        return df

    def _apply_missingness_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.ignore_high_missingness:
            return df
        missing_rate = self.data.col_missing_rate.loc[df.columns]
        keep = missing_rate < self.high_missingness_threshold
        return df.loc[:, keep]

    def _apply_column_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns:
            return df
        cols = [c for c in self.selected_columns if c in df.columns]
        return df[cols]

    def _apply_completeness_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.completeness_mode:
            return df
        completeness = self.data.col_completeness.loc[df.columns]

        if self.completeness_mode == "most":
            df = df.loc[:, completeness >= self.completeness_threshold]
            if self.max_columns_by_completeness > 0:
                completeness_filtered = self.data.col_completeness.loc[df.columns]
                idx = np.argsort(completeness_filtered)[-self.max_columns_by_completeness :]
                df = df.iloc[:, np.sort(idx)]
        elif self.completeness_mode == "least":
            df = df.loc[:, completeness <= self.completeness_threshold]
            if self.max_columns_by_completeness > 0:
                completeness_filtered = self.data.col_completeness.loc[df.columns]
                idx = np.argsort(completeness_filtered)[: self.max_columns_by_completeness]
                df = df.iloc[:, np.sort(idx)]
        return df

    def _apply_max_columns_limit(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[1] <= self.max_columns:
            return df
        return df.iloc[:, : self.max_columns]

    def _apply_ordering(self, df: pd.DataFrame) -> pd.DataFrame:
        specs = normalize_order_by(
            self.order_by,
            supports_rows=False,
            supports_data_columns=False,
        )
        for spec in reversed(specs):
            ascending = spec["ascending"]
            if spec["column"] == "__missing__":
                missing_counts = self.data.col_missing_count.loc[df.columns]
                ordered_cols = missing_counts.sort_values(ascending=ascending, kind="stable").index
                df = df.loc[:, ordered_cols]
            else:  # "__column__", the only other value the validator allows
                df = df.loc[:, sorted(df.columns, reverse=not ascending)]
        return df
