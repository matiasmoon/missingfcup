from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
from missingfcup.plots._plot import _Plot
from missingfcup.plots._selection import select_columns

# One line per row, so both are set for legibility under overlap, not to taste:
# thin enough that crossing lines stay distinguishable, faint enough that a dense
# band still shows its density.
_LINE_OPACITY = 0.4
_LINE_WIDTH = 2.0


def _format_value(value) -> str:
    """A raw cell value as it should read in a tooltip.

    Whole numbers lose their trailing ".0" and everything else keeps two decimals,
    matching the precision the rest of the package prints.
    """
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{int(value):,}" if float(value).is_integer() else f"{value:,.2f}"
    return str(value)


class _ParallelCoordinates(_Plot):
    """
    Parallel coordinates plot (ggally style).

    Columns are laid out on the x-axis. Each row is drawn as a line
    connecting its normalized values across all selected columns.
    Lines are colored by whether a designated column is missing (NA)
    or present (!NA) in that row.

    Missing values in non-color columns appear as gaps in the lines.
    """

    def __init__(
        self,
        data: MissingData,
        *,
        selected_columns: Optional[List[str]] = None,
        missing_column: Optional[str] = None,
        high_missingness_threshold: Optional[float] = None,
        max_columns: int = 0,
        sort_by: Optional[Literal["missingness", "alphabetical"]] = None,
        ascending: bool = False,
        kind: Literal["values", "missingness"] = "values",
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)
        self.selected_columns = selected_columns
        self.high_missingness_threshold = high_missingness_threshold
        self.max_columns = max_columns
        self.sort_by = sort_by
        self.ascending = ascending
        self.missing_column = missing_column
        self.kind = kind

    def _prepare_df(self) -> pd.DataFrame:
        df = self.data.data
        if self.selected_columns is not None:
            missing_cols = [col for col in self.selected_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in the DataFrame: {missing_cols}.")

        # Axis adjacency is the whole reading of this plot -- a relationship shows up
        # between neighbouring axes and nowhere else -- so the shared ordering options
        # matter more here than on a plot whose columns are only listed.
        cols = select_columns(
            self.data,
            self.selected_columns,
            high_missingness_threshold=self.high_missingness_threshold,
            sort_by=self.sort_by,
            ascending=self.ascending,
            max_columns=self.max_columns,
        )
        df = df[cols]

        if self.kind == "values":
            non_numeric = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                raise TypeError(
                    f"parallel_coordinates() requires numeric columns.\n"
                    f"Non-numeric columns found: {non_numeric}\n"
                    f"Pass only numeric columns via selected_columns=[...]"
                )

        if self.missing_column is not None:
            if self.missing_column not in self.data.data.columns:
                raise ValueError(f"missing_column '{self.missing_column}' not found")

        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize each column to [0, 1] over its observed range."""
        result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        for col in df.columns:
            if self.kind == "missingness":
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

    def _build_lines(self, norm_df: pd.DataFrame, raw_df: pd.DataFrame, mask: pd.Series):
        """
        Return flattened (x, y) arrays and per-vertex hover data for the rows
        selected by `mask`. Rows are separated by NaN; missing column values also
        become NaN (gap).

        The y values are normalised onto a shared 0-1 axis, which makes them
        unreadable as numbers, so the hover data carries the raw value instead --
        that is the whole reason this plot needs a tooltip.
        """
        rows = norm_df[mask].values.astype(float)  # (n, m)
        n, m = rows.shape
        x_positions = np.arange(m, dtype=float)

        # Append NaN column as row terminator
        y_padded = np.hstack([rows, np.full((n, 1), np.nan)])  # (n, m+1)
        x_padded = np.tile(np.append(x_positions, np.nan), n)  # n*(m+1)

        columns = list(raw_df.columns)
        selected = raw_df[mask]
        missing = self.data.mask_missing.loc[selected.index, columns]
        custom = []
        for label, values, na in zip(selected.index, selected.to_numpy(object), missing.to_numpy()):
            for column, value, is_na in zip(columns, values, na):
                shown = "NA" if is_na else _format_value(value)
                custom.append([str(label), column, shown])
            custom.append(["", "", ""])  # matches the NaN row terminator
        return x_padded, y_padded.flatten(), custom

    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()
        norm_df = self._normalize(df)
        cols = df.columns.tolist()
        m = len(cols)
        x_positions = list(range(m))

        if self.missing_column is not None:
            target_missing = self.data.mask_missing[self.missing_column]
            groups = [
                (~target_missing, f"!NA-{self.missing_column}", self.present_color),
                (target_missing, f"NA-{self.missing_column}", self.missing_color),
            ]
        else:
            # Every line is the same colour, so a legend entry would name a
            # distinction that is not drawn.
            groups = [
                (pd.Series(True, index=norm_df.index), None, self.present_color),
            ]

        fig = go.Figure()

        for mask, name, color in groups:
            if not mask.any():
                continue
            x_flat, y_flat, custom = self._build_lines(norm_df, df, mask)
            fig.add_scatter(
                x=x_flat,
                y=y_flat,
                mode="lines",
                name=name,
                showlegend=name is not None,
                line=dict(color=color, width=_LINE_WIDTH),
                opacity=_LINE_OPACITY,
                customdata=custom,
                hovertemplate=_hover.build(
                    _hover.title("row %{customdata[0]}"),
                    "%{customdata[1]}: %{customdata[2]}",
                    *([name] if name else []),
                ),
            )

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=x_positions,
                ticktext=self._truncate_labels(cols),
                tickangle=-45,
                showgrid=True,
                gridcolor="rgba(150,150,150,0.4)",
                zeroline=False,
                range=[-0.5, m - 0.5],
                title="Column",
            ),
            yaxis=dict(
                title="Missing" if self.kind == "missingness" else "Normalized value",
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
