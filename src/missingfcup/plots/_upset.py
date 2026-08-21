import warnings
from typing import List, Literal, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from missingfcup.core.missing_data import MissingData
from missingfcup.plots import _hover
from missingfcup.plots._plot import _Plot

# The membership dots are sized to the matrix, not to taste, and the line joining
# them has to read as thick as the dots so the intersection scans as one unit.
_DOT_SIZE = 12
_LINE_WIDTH = 3.0

# Grey, because it marks the absence of membership rather than a value: it is the
# lattice the red dots sit on, and any other colour would compete with them.
_EXCLUDED_DOT_COLOR = "#e0e0e0"

# Intersections grow as 2**n, so the bar count has to be capped or a wide selection
# draws an unreadable comb. Truncation loses information, so it warns rather than
# trimming quietly. Every intersection that occurs at all is worth drawing, so the
# size floor is 1 and only the empty intersection is dropped.
_MAX_INTERSECTIONS = 20
_MIN_INTERSECTION_SIZE = 1


class _Upset(_Plot):
    """
    Bar chart of missingness intersections across columns.

    Shows:
    * Intersection sizes (how many rows share each missingness combination)
    * Set sizes (total missing count per column)
    * Dot matrix indicating which columns are missing per intersection
    """

    def __init__(
        self,
        data: MissingData,
        *,
        selected_columns: Optional[List[str]] = None,
        sort_by: Optional[Literal["size"]] = "size",
        ascending: bool = False,
        measure: Literal["count", "fraction", "percentage"] = "count",
        show_values: bool = True,
        **kwargs,
    ):
        if "show_legend" not in kwargs:
            kwargs["show_legend"] = False
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.sort_by = sort_by
        self.ascending = ascending
        self.measure = measure
        self.show_values = show_values

    def _prepare_columns(self) -> List[str]:
        """The columns to draw as sets, which the caller has to name.

        Every named column is drawn. UpSet exists to compare more columns than a Venn
        can, so capping the sets here would defeat the point; the intersection cap
        already keeps the bar count readable.
        """
        if not self.selected_columns:
            # Same idea as venn(): required, but point at the columns worth comparing.
            suggestion = (
                self.data.col_missing_rate.loc[lambda s: s > 0]
                .sort_values(ascending=False)
                .index.tolist()
            )
            hint = f" Columns with missing values: {suggestion}." if suggestion else ""
            raise ValueError(f"upset() needs selected_columns: name the columns to compare.{hint}")
        df = self.data.data
        missing = [c for c in self.selected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in the DataFrame: {missing}.")
        return list(self.selected_columns)

    def _compute_intersections(self, cols: List[str]):
        mask = self.data.mask_missing[cols]
        pattern_series = mask.apply(lambda row: tuple(row.index[row]), axis=1)
        counts = pattern_series.value_counts()

        if () in counts.index:
            counts = counts.drop(())

        if _MIN_INTERSECTION_SIZE > 1:
            counts = counts.loc[lambda s: s >= _MIN_INTERSECTION_SIZE]

        if counts.empty:
            raise ValueError(
                f"None of {cols} are ever missing together, so there are no "
                f"intersections to draw. upset() compares missingness patterns, so it "
                f"needs columns that have missing values."
            )

        if self.ascending:
            counts = counts.sort_values(ascending=True)

        if len(counts) > _MAX_INTERSECTIONS:
            # Drawing the top 20 of 300 without saying so reads as "these are all the
            # patterns", which is the opposite of what the plot is for.
            warnings.warn(
                f"{len(counts)} missingness intersections found across {len(cols)} "
                f"columns; drawing the {_MAX_INTERSECTIONS} largest. Pass fewer "
                f"columns in selected_columns to see all of them.",
                UserWarning,
                stacklevel=2,
            )
            counts = counts.head(_MAX_INTERSECTIONS)

        return counts

    def _build_figure(self) -> go.Figure:
        cols = self._prepare_columns()
        intersection_counts = self._compute_intersections(cols)

        set_sizes = self.data.col_missing_count.loc[cols]
        set_sizes = set_sizes.sort_values(ascending=False)
        set_labels_full = set_sizes.index.tolist()

        cols = [c for c in set_labels_full if c in cols]

        set_labels_display = self._truncate_labels(
            set_labels_full, min_len=12, width_divisor=20, ellipsis="..."
        )

        label_map = dict(zip(set_labels_full, set_labels_display))

        subsets = list(intersection_counts.index)
        # measure means the same here as on venn(), bar() and rate(): one option
        # covering absolute counts and the two relative forms. The scaling applies to
        # both bar panels, so the intersection sizes and the set sizes stay on one
        # scale and remain comparable.
        rows = max(len(self.data.data), 1)
        if self.measure == "percentage":
            scale, value_fmt = 100.0 / rows, "{:.2f}%"
            intersection_title, set_title = "Percent of rows", "Percent of rows"
        elif self.measure == "fraction":
            scale, value_fmt = 1.0 / rows, "{:.2f}"
            intersection_title, set_title = "Fraction of rows", "Fraction of rows"
        else:
            scale, value_fmt = 1.0, "{:.0f}"
            intersection_title, set_title = "Rows", "Missing rows"

        intersection_values = [v * scale for v in intersection_counts.values.tolist()]
        set_values = [v * scale for v in set_sizes.values.tolist()]
        intersection_labels = [", ".join(s) for s in subsets]
        n_intersections = len(subsets)

        x_positions = list(range(1, n_intersections + 1))

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[None, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]],
            row_heights=[0.82, 0.18],
            column_widths=[0.33, 0.67],
            vertical_spacing=0.015,
            horizontal_spacing=0.02,
        )

        fig.add_bar(
            x=x_positions,
            y=intersection_values,
            marker_color=self.missing_color,
            text=[value_fmt.format(v) if self.show_values else None for v in intersection_values],
            textposition="outside" if self.show_values else None,
            # Same shape as venn(): name the region, then say how big it is in both
            # forms, so the reading does not depend on which measure is set.
            hovertemplate=_hover.build(
                _hover.title("%{customdata[0]}"),
                "%{customdata[1]}",
            ),
            customdata=_hover.customdata(
                intersection_labels,
                [_hover.rows_of_total(v, rows) for v in intersection_counts.values],
            ),
            row=1,
            col=2,
        )

        fig.add_bar(
            x=set_values,
            y=set_labels_display,
            orientation="h",
            marker_color=self.missing_color,
            width=0.75,
            text=[value_fmt.format(v) if self.show_values else None for v in set_values],
            textposition="outside" if self.show_values else None,
            cliponaxis=False,
            hovertemplate=_hover.build(
                _hover.title("%{customdata[0]}"),
                "NA: %{customdata[1]}",
            ),
            customdata=_hover.customdata(
                set_labels_full,
                [_hover.rows_of_total(v, rows) for v in set_sizes.values],
            ),
            row=2,
            col=1,
        )

        included_x = []
        included_y = []
        included_colors = []
        excluded_x = []
        excluded_y = []
        line_x = []
        line_y = []

        for idx, subset in enumerate(subsets, start=1):
            subset_set = set(subset)
            included_full = [label for label in set_labels_full if label in subset_set]
            excluded_full = [label for label in set_labels_full if label not in subset_set]
            included = [label_map[label] for label in included_full]
            excluded = [label_map[label] for label in excluded_full]

            for display_label in included:
                included_x.append(idx)
                included_y.append(display_label)
                included_colors.append(self.missing_color)
            for display_label in excluded:
                excluded_x.append(idx)
                excluded_y.append(display_label)

            if len(included) >= 2:
                line_x.extend([idx, idx, None])
                line_y.extend([included[0], included[-1], None])

        fig.add_scatter(
            x=excluded_x,
            y=excluded_y,
            mode="markers",
            marker=dict(size=_DOT_SIZE, color=_EXCLUDED_DOT_COLOR),
            hoverinfo="skip",
            row=2,
            col=2,
        )
        if line_x:
            fig.add_scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color=self.missing_color, width=_LINE_WIDTH),
                hoverinfo="skip",
                row=2,
                col=2,
            )
        fig.add_scatter(
            x=included_x,
            y=included_y,
            mode="markers",
            marker=dict(
                size=_DOT_SIZE,
                color=included_colors if included_colors else self.missing_color,
            ),
            hovertemplate=_hover.build(_hover.title("%{customdata}")),
            customdata=[intersection_labels[x - 1] for x in included_x],
            row=2,
            col=2,
        )

        fig.update_xaxes(showticklabels=False, row=1, col=2)
        fig.update_xaxes(showticklabels=False, row=2, col=2)
        fig.update_xaxes(
            tickfont=dict(size=14),
            row=2,
            col=1,
        )
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=set_labels_display,
            autorange="reversed",
            showticklabels=False,
            row=2,
            col=1,
        )
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=set_labels_display,
            autorange="reversed",
            showticklabels=False,
            row=2,
            col=2,
        )
        fig.update_yaxes(
            showticklabels=True,
            side="left",
            ticks="",
            ticklabelstandoff=12,
            tickfont=dict(size=15),
            row=2,
            col=1,
        )

        if n_intersections > 0:
            x_range = [0.5, n_intersections + 0.5]
            fig.update_xaxes(range=x_range, row=1, col=2)
            fig.update_xaxes(range=x_range, row=2, col=2)

        # measure changes what the bars mean, so the axes have to say which it is.
        # Nothing labelled them before, which left "4" and "0.20" looking alike.
        fig.update_yaxes(title_text=intersection_title, row=1, col=2)
        fig.update_xaxes(title_text=set_title, row=2, col=1)

        max_val = max(intersection_values) if intersection_values else 0
        if max_val > 0:
            fig.update_yaxes(range=[0, max_val * 1.25], row=1, col=2)

        max_set = max(set_sizes.values.tolist()) if len(set_sizes) else 0
        if max_set > 0:
            fig.update_xaxes(range=[0, max_set * 1.18], row=2, col=1)

        self._apply_base_layout(fig)
        fig.update_layout(
            margin=dict(t=70, b=70, l=70, r=60),
            bargap=0.35,
            hoverlabel=dict(font_size=14),
        )

        return fig
