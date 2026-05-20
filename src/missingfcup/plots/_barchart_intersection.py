import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Literal

from missingfcup.plots._plot import _Plot
from missingfcup.core.missing_data import MissingData


class _BarchartIntersection(_Plot):
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
        selected_columns: Optional[List[str]] = None,
        max_sets: int = 3,
        max_intersections: int = 20,
        min_intersection_size: int = 1,
        order: Literal["desc", "asc"] = "desc",
        show_values: bool = True,
        matrix_dot_size: int = 14,
        matrix_line_width: int = 4,
        excluded_dot_color: str = "#e0e0e0",
        highlight_columns: Optional[List[str]] = None,
        highlight_color: Optional[str] = None,
        **kwargs,
    ):
        if "show_legend" not in kwargs:
            kwargs["show_legend"] = False
        super().__init__(data=data, **kwargs)

        self.selected_columns = selected_columns
        self.max_sets = max_sets
        self.max_intersections = max_intersections
        self.min_intersection_size = min_intersection_size
        self.order = order
        self.show_values = show_values
        self.matrix_dot_size = matrix_dot_size
        self.matrix_line_width = matrix_line_width
        self.excluded_dot_color = excluded_dot_color
        self.highlight_columns = highlight_columns
        self.highlight_color = highlight_color

    # ------------------------------------------------------------------
    # Data prep
    # ------------------------------------------------------------------
    def _prepare_columns(self) -> List[str]:
        df = self.data.data
        if self.selected_columns:
            cols = [c for c in self.selected_columns if c in df.columns]
            if not cols:
                raise ValueError("No selected columns found in the DataFrame.")
        else:
            missing_rate = self.data.col_missing_rate
            cols = (
                missing_rate.loc[lambda s: s > 0]
                .sort_values(ascending=False)
                .index.tolist()
            )

        if not cols:
            raise ValueError("barchart_intersection requires at least one column with missing values.")

        if self.max_sets > 0:
            cols = cols[: self.max_sets]

        return cols

    def _compute_intersections(self, cols: List[str]):
        mask = self.data.mask_missing[cols]
        pattern_series = mask.apply(lambda row: tuple(row.index[row]), axis=1)
        counts = pattern_series.value_counts()

        if () in counts.index:
            counts = counts.drop(())

        if self.min_intersection_size > 1:
            counts = counts.loc[lambda s: s >= self.min_intersection_size]

        if counts.empty:
            raise ValueError("No missingness intersections found for selected columns.")

        if self.order == "asc":
            counts = counts.sort_values(ascending=True)

        if self.max_intersections > 0:
            counts = counts.head(self.max_intersections)

        return counts

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------
    def _build_figure(self) -> go.Figure:
        cols = self._prepare_columns()
        intersection_counts = self._compute_intersections(cols)

        set_sizes = self.data.col_missing_count.loc[cols]
        set_sizes = set_sizes.sort_values(ascending=False)
        set_labels_full = set_sizes.index.tolist()

        cols = [c for c in set_labels_full if c in cols]

        def resolved_max_label_length() -> int:
            if self.max_label_length > 0:
                return self.max_label_length
            return max(12, int(self.width / 20))

        max_len = resolved_max_label_length()

        def truncate_label(label: str) -> str:
            if max_len <= 0 or len(label) <= max_len:
                return label
            return label[: max_len - 3] + "..."

        set_labels_display = [truncate_label(l) for l in set_labels_full]

        if len(set(set_labels_display)) < len(set_labels_display):
            counts_seen = {}
            adjusted = []
            for label in set_labels_display:
                counts_seen[label] = counts_seen.get(label, 0) + 1
                idx = counts_seen[label]
                suffix = " ..."
                base = label
                if max_len > len(suffix):
                    base = label[: max_len - len(suffix)]
                adjusted.append(base + suffix + (" " * (idx - 1)))
            set_labels_display = adjusted

        label_map = dict(zip(set_labels_full, set_labels_display))

        subsets = list(intersection_counts.index)
        intersection_values = intersection_counts.values.tolist()
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
            text=[str(v) if self.show_values else None for v in intersection_values],
            textposition="outside" if self.show_values else None,
            hovertemplate=(
                "<b>Missing columns</b>: %{customdata}<br>"
                "<b>Rows</b>: %{y}<extra></extra>"
            ),
            customdata=intersection_labels,
            row=1,
            col=2,
        )

        bar_colors = None
        if self.highlight_columns:
            highlight_set = set(self.highlight_columns)
            bar_colors = [
                self.highlight_color if label in highlight_set else self.missing_color
                for label in set_labels_full
            ]

        fig.add_bar(
            x=set_sizes.values.tolist(),
            y=set_labels_display,
            orientation="h",
            marker_color=bar_colors if bar_colors is not None else self.missing_color,
            width=0.75,
            text=[str(int(v)) if self.show_values else None for v in set_sizes.values],
            textposition="outside" if self.show_values else None,
            textfont=dict(size=16),
            cliponaxis=False,
            hovertemplate=(
                "<b>Column</b>: %{y}<br>"
                "<b>Missing</b>: %{x}<extra></extra>"
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

            for full_label, display_label in zip(included_full, included):
                included_x.append(idx)
                included_y.append(display_label)
                if self.highlight_columns and full_label in self.highlight_columns:
                    included_colors.append(self.highlight_color or self.missing_color)
                else:
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
            marker=dict(size=self.matrix_dot_size, color=self.excluded_dot_color),
            hoverinfo="skip",
            row=2,
            col=2,
        )
        if line_x:
            fig.add_scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color=self.missing_color, width=self.matrix_line_width),
                hoverinfo="skip",
                row=2,
                col=2,
            )
        fig.add_scatter(
            x=included_x,
            y=included_y,
            mode="markers",
            marker=dict(
                size=self.matrix_dot_size,
                color=included_colors if included_colors else self.missing_color,
            ),
            hovertemplate="<b>Columns</b>: %{customdata}<extra></extra>",
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
