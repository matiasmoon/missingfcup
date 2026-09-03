import warnings
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
from missingfcup.plots._color import hex_to_rgba
from missingfcup.plots._plot import _Plot
from missingfcup.plots._selection import select_columns

# Most y-axis ticks to draw; more than this and the labels overlap.
_MAX_Y_TICKS = 12

# Fixed presentation for the border marking the column ``sort_by`` names. Must read
# as annotation, not data, so it is deliberately not one of the plot's colours and
# not tunable.
_SORT_BORDER_COLOR = "#1f77b4"
_SORT_BORDER_WIDTH = 2
_SORT_BORDER_FILL_OPACITY = 0.08

# Cell separation in pixels. Asymmetric on purpose: a vertical seam separates
# columns, the unit being compared, while a horizontal one between rows would only
# shred a tall matrix into stripes.
_XGAP = 1
_YGAP = 0


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
        high_missingness_threshold: Optional[float] = None,
        max_columns: int = 0,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        sort_categories: Optional[Sequence] = None,
        **kwargs,
    ):
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.high_missingness_threshold = high_missingness_threshold
        self.max_columns = max_columns
        self.sort_by = sort_by
        self.ascending = ascending
        self.sort_categories = sort_categories

    @property
    def _sort_column(self) -> Optional[str]:
        """The data column ``sort_by`` names, if it names one.

        ``sort_by`` is either one of the two keywords, which order the columns, or the
        name of a column in the frame, which orders the rows by that column's values.
        """
        if self.sort_by in (None, "missingness", "alphabetical"):
            return None
        return self.sort_by

    def _prepare_df(self) -> pd.DataFrame:
        row_column = self._sort_column
        cols = select_columns(
            self.data,
            self.selected_columns or None,
            high_missingness_threshold=self.high_missingness_threshold,
            # A data column sorts rows, so the columns keep the frame's own order.
            sort_by=None if row_column else self.sort_by,
            ascending=self.ascending,
            max_columns=self.max_columns,
        )
        df = self.data.data[cols].copy()

        if self.sort_categories is not None and row_column is None:
            # Otherwise the order would be accepted and quietly ignored: nothing is
            # sorting the rows for it to apply to.
            raise ValueError(
                "sort_categories needs sort_by to name a column whose values it "
                f"orders. Got sort_by={self.sort_by!r}."
            )

        if row_column is not None:
            if row_column not in df.columns:
                raise ValueError(
                    f"sort_by={row_column!r} is not a drawn column. Use 'missingness', "
                    f"'alphabetical', or one of {list(df.columns)}."
                )
            df = df.loc[self._row_order(df[row_column])]
            # Pin it left so the ordering it produced is easy to read off.
            df = df[[row_column] + [c for c in df.columns if c != row_column]]

        return df

    def _row_order(self, values: pd.Series) -> pd.Index:
        """The row index in the order the rows should be drawn.

        Without ``sort_categories`` this is just ``sort_values``, so the column's
        dtype decides. With it, the caller's sequence decides instead, which is the
        only way to order a nominal categorical meaningfully.
        """
        if self.sort_categories is None:
            return values.sort_values(
                ascending=self.ascending, kind="stable", na_position="last"
            ).index

        declared = list(self.sort_categories)
        rank = {value: position for position, value in enumerate(declared)}
        present = set(values.dropna().unique())
        if not present & set(declared):
            raise ValueError(
                f"None of sort_categories {declared} appear in {values.name!r}. "
                f"It holds {sorted(present, key=repr)}."
            )

        absent = [value for value in declared if value not in present]
        if absent:
            # Not an error: the same order is worth reusing across datasets, and a
            # value this frame happens to lack is fine. But a typo looks identical
            # from here and would otherwise reorder nothing and say nothing.
            warnings.warn(
                f"sort_categories named {absent}, which {values.name!r} does not "
                f"contain. It holds {sorted(present, key=repr)}. Those entries order "
                f"nothing.",
                UserWarning,
                stacklevel=2,
            )

        # Three buckets, always in this order: values the caller named, then
        # anything they did not name, then the missing ones. ``ascending`` is not
        # consulted -- the sequence already says which value is first and which is
        # last, so honouring a direction on top would draw the reverse of what the
        # caller wrote. Reverse the sequence to reverse the plot.
        def bucket(value) -> int:
            if pd.isna(value):
                return 2
            return 0 if value in rank else 1

        order = pd.DataFrame(
            {
                "bucket": [bucket(v) for v in values],
                "rank": [rank.get(v, 0) for v in values],
            },
            index=values.index,
        )
        return order.sort_values(["bucket", "rank"], kind="stable").index

    @property
    def row_labels(self) -> List[str]:
        """The row labels in drawn order, one per row.

        The y axis thins its ticks to stay readable, so the rendered labels are a
        subset. This is the full sequence: the value of ``sort_by`` for each row, in
        the order the matrix draws them, or the row index when no column was named.
        """
        self.fig  # noqa: B018 -- builds and caches the figure if it has not been built
        return self._row_labels

    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()

        # Reuse cached missingness mask and align to the filtered frame
        mask = self.data.mask_missing.loc[df.index, df.columns]

        z = (~mask).astype(int).to_numpy()
        # Four stops, not two: a binary variable has no in-between, so the bar
        # must show two solid blocks rather than interpolate between them.
        colorscale = [
            [0.0, self.missing_color],
            [0.5, self.missing_color],
            [0.5, self.present_color],
            [1.0, self.present_color],
        ]
        colorbar_ticks = ["NA", "!NA"]

        x_labels = self._truncate_labels(df.columns.tolist())

        x_positions = list(range(len(df.columns)))

        # When sort_by names a column the rows are in that column's order, so the
        # axis labels it: a row's own value says where it sits. Otherwise there is
        # nothing to label rows with but their index.
        sort_column = self._sort_column
        if sort_column and sort_column in df.columns:

            def _fmt(v):
                if pd.isna(v):
                    return "NaN"
                if isinstance(v, float) and v == int(v):
                    return str(int(v))
                return str(v)

            y_labels = [_fmt(v) for v in df[sort_column].values]
        else:
            y_labels = [str(i) for i in df.index]

        # The labels are not the coordinates. A sort column repeats its values, and
        # go.Heatmap treats a repeated y as the same row, so every row sharing a value
        # would collapse onto one and overplot. Positions stay unique, exactly as the
        # x axis already does, and the labels are attached as ticks below.
        y_positions = list(range(len(df)))

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x_positions,
                y=y_positions,
                colorscale=colorscale,
                zmin=0,
                zmax=1,
                xgap=_XGAP,
                ygap=_YGAP,
                showscale=self.show_legend,
                colorbar=dict(
                    tickvals=[0.25, 0.75],  # centre of each block, not its edge
                    ticktext=colorbar_ticks,
                    len=0.5,
                    thickness=14,
                )
                if self.show_legend
                else None,
                hovertemplate=_hover.build(
                    _hover.title("%{customdata[0]}  \u00b7  row %{y}"),
                    "%{text}: %{customdata[1]}",
                ),
                # Derived from the mask, not from z, so the label cannot drift out
                # of step with how z happens to be encoded.
                text=[["NA" if v else "!NA" for v in row] for row in mask.to_numpy()],
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

        # tickmode="array" draws every tick it is given, so the labels have to be
        # thinned here or they overlap. The full ordered sequence stays available on
        # the plot object as ``row_labels``.
        if sort_column and sort_column in df.columns:
            seen = set()
            y_tickvals, y_ticktext = [], []
            for pos, label in zip(y_positions, y_labels):
                if label not in seen:
                    seen.add(label)
                    y_tickvals.append(pos)
                    y_ticktext.append(label)
            if len(y_tickvals) > _MAX_Y_TICKS:
                step = len(y_tickvals) // _MAX_Y_TICKS + 1
                y_tickvals, y_ticktext = y_tickvals[::step], y_ticktext[::step]
        else:
            step = max(1, len(y_positions) // _MAX_Y_TICKS)
            y_tickvals, y_ticktext = y_positions[::step], y_labels[::step]

        self._row_labels = list(y_labels)
        fig.update_yaxes(tickmode="array", tickvals=y_tickvals, ticktext=y_ticktext)
        # When sort_by names a column the y labels are that column's values, so the
        # axis is named after it; otherwise rows are only identified by index.
        fig.update_xaxes(title_text="Column")
        fig.update_yaxes(
            title_text=sort_column if sort_column and sort_column in df.columns else "Row",
            title_standoff=15,
        )

        # Outline order-by columns (if any) to make ordering visible
        order_cols = [c for c in [self._sort_column] if c and c in df.columns]

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
                            color=_SORT_BORDER_COLOR,
                            width=_SORT_BORDER_WIDTH,
                        ),
                        fillcolor=hex_to_rgba(_SORT_BORDER_COLOR, _SORT_BORDER_FILL_OPACITY),
                        layer="above",
                    )
                )
            fig.update_layout(shapes=shapes)

        self._apply_base_layout(fig)

        return fig
