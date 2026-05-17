import plotly.graph_objects as go
from typing import Optional, Literal, List, Dict
import pandas as pd
import numpy as np

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData


class _BarchartRate(_Plot):
    """
    Bar chart of missing rate per column.

    Shows the fraction or percentage of missing values for each column.
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
        scale: Literal["fraction", "percentage"] = "fraction",
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
        self.scale = scale

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

    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()
        columns = df.columns.tolist()

        rates = self.data.col_missing_rate.loc[columns]

        if self.scale == "percentage":
            values = rates * 100
            y_title = "Missing (%)"
            text_vals = [f"{v:.1f}%" for v in values]
        else:
            values = rates
            y_title = "Missing rate"
            text_vals = [f"{v:.2f}" for v in values]

        fig = go.Figure()

        fig.add_bar(
            x=columns if self.orientation == "vertical" else values,
            y=values if self.orientation == "vertical" else columns,
            name="Missing",
            marker_color=self.missing_color,
            text=text_vals if self.show_values else None,
            textposition="auto" if self.show_values else None,
        )

        first_col = columns[0] if columns else ""
        if self.orientation == "vertical":
            fig.update_xaxes(tickangle=-45, title_text=first_col)
            fig.update_yaxes(title_text=y_title)
        else:
            fig.update_xaxes(title_text=y_title)
            fig.update_yaxes(title_text=first_col)
        self._apply_base_layout(fig)
        return fig
