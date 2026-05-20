import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, List

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData


class _ParallelCoordinates(_Plot):
    """
    Parallel coordinates plot (ggally style).

    Columns are laid out on the x-axis. Each row is drawn as a line
    connecting its normalized values across all selected columns.
    Lines are coloured by whether a designated column is missing (NA)
    or present (!NA) in that row.

    Missing values in non-color columns appear as gaps in the lines.
    """

    def __init__(
        self,
        data: MissingData,
        *,
        selected_columns: Optional[List[str]] = None,
        missingness_color_column: Optional[str] = None,
        line_opacity: float = 0.4,
        line_width: float = 1.0,
        missingness_only: bool = False,
        normalize: bool = True,
        **kwargs,
    ):
        legend_title = kwargs.pop(
            "legend_title",
            f"{missingness_color_column}_NA" if missingness_color_column else "Status",
        )
        super().__init__(data=data, legend_title=legend_title, **kwargs)
        self.selected_columns = selected_columns
        self.missingness_color_column = missingness_color_column
        self.line_opacity = line_opacity
        self.line_width = line_width
        self.missingness_only = missingness_only

    def _prepare_df(self) -> pd.DataFrame:
        df = self.data.data
        if self.selected_columns is None:
            selected = df.columns.tolist()
        else:
            selected = self.selected_columns

        missing_cols = [col for col in selected if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")

        df = df[selected]

        if not self.missingness_only:
            non_numeric = [
                col for col in df.columns
                if not pd.api.types.is_numeric_dtype(df[col])
            ]
            if non_numeric:
                raise TypeError(
                    f"parallel_coordinates() requires numeric columns.\n"
                    f"Non-numeric columns found: {non_numeric}\n"
                    f"Pass only numeric columns via selected_columns=[...]"
                )

        if self.missingness_color_column is not None:
            if self.missingness_color_column not in self.data.data.columns:
                raise ValueError(
                    f"missingness_color_column "
                    f"'{self.missingness_color_column}' not found"
                )

        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize each column to [0, 1] over its observed range."""
        result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        for col in df.columns:
            if self.missingness_only:
                result[col] = self.data.mask_missing[col].astype(float)
            else:
                s = df[col].dropna()
                if s.empty:
                    result[col] = np.nan
                    continue
                min_val = float(s.min())
                max_val = float(s.max())
                span = max_val - min_val or 1.0
                result[col] = (df[col] - min_val) / span
        return result

    def _build_lines(
        self, norm_df: pd.DataFrame, mask: pd.Series
    ):
        """
        Return flattened (x, y) arrays for all rows selected by `mask`.
        Rows are separated by NaN; missing column values also become NaN (gap).
        """
        rows = norm_df[mask].values.astype(float)   # (n, m)
        n, m = rows.shape
        x_positions = np.arange(m, dtype=float)

        # Append NaN column as row terminator
        y_padded = np.hstack([rows, np.full((n, 1), np.nan)])          # (n, m+1)
        x_padded = np.tile(np.append(x_positions, np.nan), n)           # n*(m+1)
        return x_padded, y_padded.flatten()

    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()
        norm_df = self._normalize(df)
        cols = df.columns.tolist()
        m = len(cols)
        x_positions = list(range(m))

        if self.missingness_color_column is not None:
            target_missing = self.data.mask_missing[self.missingness_color_column]
            groups = [
                (~target_missing, "!NA", self.present_color),
                (target_missing,  "NA",  self.missing_color),
            ]
        else:
            groups = [
                (pd.Series(True, index=norm_df.index), "values", self.present_color),
            ]

        fig = go.Figure()

        for mask, name, color in groups:
            if not mask.any():
                continue
            x_flat, y_flat = self._build_lines(norm_df, mask)
            fig.add_scatter(
                x=x_flat,
                y=y_flat,
                mode="lines",
                name=name,
                line=dict(color=color, width=self.line_width),
                opacity=self.line_opacity,
                hoverinfo="skip",
            )

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=x_positions,
                ticktext=cols,
                tickangle=-45,
                showgrid=True,
                gridcolor="rgba(150,150,150,0.4)",
                zeroline=False,
                range=[-0.5, m - 0.5],
                title=cols[0] if cols else "",
            ),
            yaxis=dict(
                title="Normalized value",
                range=[-0.05, 1.08],
                tickmode="array",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0.00", "0.25", "0.50", "0.75", "1.00"],
                showgrid=True,
                zeroline=False,
            ),
            dragmode="pan",
        )

        self._apply_base_layout(fig)
        return fig
