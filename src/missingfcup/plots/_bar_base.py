from typing import List, Literal, Optional

import pandas as pd

from missingfcup.core.missing_data import MissingData
from missingfcup.plots._plot import _Plot
from missingfcup.plots._selection import select_columns


class _BarBase(_Plot):
    """Shared column picking for the per-column bar charts (count and rate).

    Both bars select, filter and order their columns the same way. Only what they draw
    at the end differs, so a subclass just implements ``_build_figure`` and may add its
    own options (``value``/``show_both`` for counts, ``measure`` for rate).
    """

    def __init__(
        self,
        data: MissingData,
        *,
        selected_columns: Optional[List[str]] = None,
        high_missingness_threshold: Optional[float] = None,
        max_columns: int = 0,
        sort_by: Optional[Literal["missingness", "alphabetical"]] = None,
        ascending: bool = False,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        show_values: bool = True,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.high_missingness_threshold = high_missingness_threshold
        self.max_columns = max_columns
        self.sort_by = sort_by
        self.ascending = ascending
        self.orientation = orientation
        self.show_values = show_values

    def _prepare_df(self) -> pd.DataFrame:
        # Every step is the shared helper's, so the bars filter, order and cap their
        # columns exactly as the other column plots do, including raising when
        # nothing the caller named survives.
        cols = select_columns(
            self.data,
            self.selected_columns,
            high_missingness_threshold=self.high_missingness_threshold,
            sort_by=self.sort_by,
            ascending=self.ascending,
            max_columns=self.max_columns,
        )
        return self.data.data.loc[:, cols]
