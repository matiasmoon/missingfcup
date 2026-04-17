import plotly.graph_objects as go
from typing import Optional, Literal, List, Dict
import pandas as pd
import numpy as np

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData

class _MissingCountBarChart(_Plot):
    """
    Bar chart of missing and/or present counts per column.
    """

    def __init__(
        self,
        data: MissingData,
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
        self.value = value
        self.show_both = show_both

    # ------------------------------------------------------------------
    # Data prep
    # ------------------------------------------------------------------
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
        if not self.order_by:
            return df

        for spec in reversed(self.order_by):
            col = spec["column"]
            ascending = spec.get("numeric_order", "asc") == "asc"
            if "ascending" in spec:
                ascending = bool(spec["ascending"])

            if col == "__missing__":
                missing_counts = self.data.col_missing_count.loc[df.columns]
                ordered_cols = missing_counts.sort_values(
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

    def _compute_missing_values(self, df: pd.DataFrame) -> pd.Series:
        return self.data.col_missing_count.loc[df.columns]

    def _compute_present_values(self, df: pd.DataFrame) -> pd.Series:
        return self.data.col_present_count.loc[df.columns]

    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()
        missing_values = self._compute_missing_values(df)
        present_values = self._compute_present_values(df)
        columns = missing_values.index.tolist()
        value_title = "Count"

        fig = go.Figure()

        if self.show_both:
            fig.add_bar(
                x=columns if self.orientation == "vertical" else missing_values,
                y=missing_values if self.orientation == "vertical" else columns,
                name="Missing",
                marker_color=self.missing_color,
                text=[f"{int(v)}" if self.show_values else None for v in missing_values],
                textposition="auto" if self.show_values else None,
            )
            fig.add_bar(
                x=columns if self.orientation == "vertical" else present_values,
                y=present_values if self.orientation == "vertical" else columns,
                name="Present",
                marker_color=self.present_color,
                text=[f"{int(v)}" if self.show_values else None for v in present_values],
                textposition="auto" if self.show_values else None,
            )
            fig.update_layout(barmode="stack")
        else:
            values = present_values if self.value == "present" else missing_values
            name = "Present" if self.value == "present" else "Missing"
            color = self.present_color if self.value == "present" else self.missing_color
            fig.add_bar(
                x=columns if self.orientation == "vertical" else values,
                y=values if self.orientation == "vertical" else columns,
                name=name,
                marker_color=color,
                text=[f"{int(v)}" if self.show_values else None for v in values],
                textposition="auto" if self.show_values else None,
            )

        if self.orientation == "vertical":
            fig.update_xaxes(tickangle=-45)
        self._apply_base_layout(fig)
        return fig
