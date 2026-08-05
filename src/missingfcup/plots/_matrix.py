import warnings
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots._color import hex_to_rgba
from missingfcup.plots._plot import _Plot
from missingfcup.plots._selection import select_columns


class _Matrix(_Plot):
    """
    Interactive binary missingness matrix (nullity matrix).

    Rows = observations
    Columns = variables
    Cell color indicates missing vs present
    """

    def __init__(
        self,
        data: MissingData,
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
        order_by_border_color: str = "#1f77b4",
        order_by_border_width: int = 5,
        order_by_y_labels: bool = True,
        **kwargs,
    ):
        super().__init__(
            data=data,
            legend_title="Status" if show_colorscale else None,
            **kwargs,
        )

        self.selected_columns = selected_columns
        self.ignore_high_missingness = ignore_high_missingness
        self.high_missingness_threshold = high_missingness_threshold
        self.max_columns = max_columns
        self.order_by = order_by

        self.show_colorscale = show_colorscale
        self.group_by_mode = group_by_mode
        self.xgap = xgap
        self.ygap = ygap
        self.order_by_border_color = order_by_border_color
        self.order_by_border_width = order_by_border_width
        self.order_by_y_labels = order_by_y_labels

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _prepare_df(self) -> pd.DataFrame:
        cols = select_columns(
            self.data,
            self.selected_columns or None,
            ignore_high_missingness=self.ignore_high_missingness,
            high_missingness_threshold=self.high_missingness_threshold,
            max_columns=self.max_columns,
        )
        df = self.data.data[cols].copy()

        if self.order_by:
            df = self._apply_ordering(df)
            df = self._pin_order_columns_left(df)

        return df

    def _pin_order_columns_left(self, df: pd.DataFrame) -> pd.DataFrame:
        order_cols: List[str] = []
        for spec in self.order_by or []:
            col = spec.get("column")
            if not col or col in ("__missing__", "__column__", "__row__"):
                continue
            if col in df.columns and col not in order_cols:
                order_cols.append(col)

        if not order_cols:
            return df

        remaining = [c for c in df.columns if c not in order_cols]
        return df[order_cols + remaining]

    def _apply_ordering(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(self.order_by) > 2:
            raise ValueError("order_by supports at most 2 specifications.")

        for spec in reversed(self.order_by):
            axis = spec.get("axis", "columns")
            ascending = spec.get("ascending", True)
            col = spec.get("column")
            spec_type = spec.get("type", "numeric")

            if axis == "columns":
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
            else:
                if col == "__missing__":
                    missing_counts = df.isna().sum(axis=1)
                    df = df.loc[
                        missing_counts.sort_values(ascending=ascending, kind="stable").index
                    ]
                    continue
                if col == "__row__":
                    df = df.sort_index(ascending=ascending)
                    continue

            if col in df.columns:
                if spec_type == "numeric":
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        warnings.warn(
                            f"Order spec expects numeric column '{col}', "
                            "but data is non-numeric. Using default ascending/descending.",
                            UserWarning,
                            stacklevel=2,
                        )
                    df = df.sort_values(
                        col,
                        ascending=ascending,
                        kind="stable",
                        na_position="last",
                    )
                elif spec_type == "categorical":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        warnings.warn(
                            f"Order spec expects categorical column '{col}', "
                            "but data is numeric. Using default ascending/descending.",
                            UserWarning,
                            stacklevel=2,
                        )
                        df = df.sort_values(
                            col,
                            ascending=ascending,
                            kind="stable",
                            na_position="last",
                        )
                    else:
                        category_order = spec.get("category_order")
                        if category_order is not None:
                            cat = pd.Categorical(df[col], categories=category_order, ordered=True)
                            df = df.assign(**{col: cat}).sort_values(
                                col,
                                kind="stable",
                                na_position="last",
                            )
                        else:
                            df = df.sort_values(
                                col,
                                ascending=ascending,
                                kind="stable",
                                na_position="last",
                            )

        return df

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()

        # Reuse cached missingness mask and align to the filtered frame
        mask = self.data.mask_missing.loc[df.index, df.columns]

        if self.group_by_mode == "binary":
            z = (~mask).astype(int).to_numpy()
            colorscale = [
                [0.0, self.missing_color],
                [1.0, self.present_color],
            ]
            colorbar_ticks = ["Missing", "Present"]
        else:
            z = mask.astype(int).to_numpy()
            colorscale = [
                [0.0, self.present_color],
                [1.0, self.missing_color],
            ]
            colorbar_ticks = ["Present", "Missing"]

        x_labels = self._truncate_labels(df.columns.tolist())

        x_positions = list(range(len(df.columns)))

        if self.order_by and self.order_by_y_labels:
            _primary_col = next(
                (
                    spec.get("column")
                    for spec in self.order_by
                    if spec.get("column") not in (None, "__missing__", "__column__", "__row__")
                ),
                None,
            )
            if _primary_col and _primary_col in df.columns:

                def _fmt(v):
                    if pd.isna(v):
                        return "NaN"
                    if isinstance(v, float) and v == int(v):
                        return str(int(v))
                    return str(v)

                y_labels = [_fmt(v) for v in df[_primary_col].values]
            else:
                y_labels = [str(i) for i in df.index]
        else:
            y_labels = [str(i) for i in df.index]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x_positions,
                y=y_labels,
                colorscale=colorscale,
                zmin=0,
                zmax=1,
                xgap=self.xgap,
                ygap=self.ygap,
                showscale=self.show_colorscale,
                colorbar=dict(
                    tickvals=[0, 1],
                    ticktext=colorbar_ticks,
                    len=0.4,
                )
                if self.show_colorscale
                else None,
                hovertemplate=(
                    "<b>Row</b>: %{y}<br>"
                    "<b>Column</b>: %{customdata[0]}<br>"
                    "<b>Status</b>: %{text}<br>"
                    "<b>Value</b>: %{customdata[1]}<extra></extra>"
                ),
                text=[["Present" if v == 1 else "Missing" for v in row] for row in z],
                customdata=np.stack(
                    [
                        np.array([[col for col in df.columns] for _ in range(len(df))]),
                        np.where(
                            mask.to_numpy(),
                            "NaN",
                            df.to_numpy(dtype=object),
                        ),
                    ],
                    axis=-1,
                ),
            )
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=x_positions,
            ticktext=x_labels,
            tickangle=-45,
        )
        fig.update_yaxes(
            title_standoff=15,
        )

        # Outline order-by columns (if any) to make ordering visible
        order_cols = []
        for spec in self.order_by or []:
            col = spec.get("column")
            if col and col in df.columns and col not in order_cols:
                order_cols.append(col)

        if order_cols:
            shapes = []
            col_positions = {col: idx for idx, col in enumerate(df.columns)}
            for col in order_cols:
                pos = col_positions[col]
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=pos - 0.5,
                        x1=pos + 0.5,
                        y0=0,
                        y1=1,
                        line=dict(
                            color=self.order_by_border_color,
                            width=self.order_by_border_width,
                        ),
                        fillcolor=hex_to_rgba(self.order_by_border_color, 0.08),
                        layer="above",
                    )
                )
            fig.update_layout(shapes=shapes)

        self._apply_base_layout(fig)

        return fig
