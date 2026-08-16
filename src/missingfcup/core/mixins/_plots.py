from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, cast

if TYPE_CHECKING:
    from missingfcup.core.missing_data import MissingData
    from missingfcup.plots._boxplot import _Boxplot
    from missingfcup.plots._dendrogram import _Dendrogram
    from missingfcup.plots._density import _Density
    from missingfcup.plots._matrix import _Matrix
    from missingfcup.plots._parallel_coordinates import _ParallelCoordinates
    from missingfcup.plots._plot import _Plot
    from missingfcup.plots._rate import _Rate
    from missingfcup.plots._scatterplot import _Scatterplot
    from missingfcup.plots._totals import _Totals
    from missingfcup.plots._upset import _Upset
    from missingfcup.plots._venn import _Venn


def _md(mixin: "_MissingDataPlotMixin") -> "MissingData":
    """Narrow the mixin to the concrete class it is always part of.

    The plot classes take a ``MissingData``, and this mixin exists only to keep the
    factories in their own file, so ``self`` is always one. Stating that here lets a
    type checker verify every construction below.
    """
    return cast("MissingData", mixin)


class _MissingDataPlotMixin:
    """
    All plot factory methods for MissingData.

    Each method constructs and returns a plot object. Imports are lazy
    (inside the method body) to avoid circular imports between core and plots.
    """

    def bar(
        self,
        *,
        measure: Literal["count", "rate"] = "count",
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
        scale: Literal["fraction", "percentage"] = "fraction",
        **kwargs,
    ) -> "_Plot":
        """
        Create a per-column bar chart of missingness.

        ``measure="count"`` (default) plots missing (or present) counts per
        column; ``value`` and ``show_both`` apply. ``measure="rate"`` plots the
        missing rate per column; ``scale`` applies.
        """
        shared = dict(
            data=_md(self),
            selected_columns=selected_columns,
            ignore_high_missingness=ignore_high_missingness,
            high_missingness_threshold=high_missingness_threshold,
            completeness_mode=completeness_mode,
            completeness_threshold=completeness_threshold,
            max_columns_by_completeness=max_columns_by_completeness,
            max_columns=max_columns,
            order_by=order_by,
            orientation=orientation,
            show_values=show_values,
        )
        if measure == "count":
            from missingfcup.plots._bar_count import _BarCount

            return _BarCount(**shared, value=value, show_both=show_both, **kwargs)
        if measure == "rate":
            from missingfcup.plots._bar_rate import _BarRate

            return _BarRate(**shared, scale=scale, **kwargs)
        raise ValueError(f"measure must be 'count' or 'rate', got {measure!r}")

    def totals(self, **kwargs) -> "_Totals":
        """Create a bar chart showing total present vs missing cell counts across the dataset."""
        from missingfcup.plots._totals import _Totals

        return _Totals(data=_md(self), **kwargs)

    def matrix(
        self,
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
        order_by_border_width: int = 2,
        order_by_y_labels: bool = True,
        **kwargs,
    ) -> "_Matrix":
        """Create an interactive binary missingness matrix (nullity matrix)."""
        from missingfcup.plots._matrix import _Matrix

        return _Matrix(
            data=_md(self),
            selected_columns=selected_columns,
            ignore_high_missingness=ignore_high_missingness,
            high_missingness_threshold=high_missingness_threshold,
            max_columns=max_columns,
            order_by=order_by,
            show_colorscale=show_colorscale,
            group_by_mode=group_by_mode,
            xgap=xgap,
            ygap=ygap,
            order_by_border_color=order_by_border_color,
            order_by_border_width=order_by_border_width,
            order_by_y_labels=order_by_y_labels,
            **kwargs,
        )

    def scatterplot(
        self,
        x: str,
        y: str,
        *,
        point_size: int = 8,
        axis_padding: float = 0.1,
        missingness_color_column: Optional[str] = None,
        point_opacity: float = 0.7,
        jitter: float = 0.02,
        missing_jitter: float = 0.5,
        jitter_seed: int = 42,
        xaxis_range: Optional[list] = None,
        yaxis_range: Optional[list] = None,
        **kwargs,
    ) -> "_Scatterplot":
        """Create a scatter plot that keeps missing values visible via axis offsets."""
        from missingfcup.plots._scatterplot import _Scatterplot

        return _Scatterplot(
            data=_md(self),
            x=x,
            y=y,
            point_size=point_size,
            axis_padding=axis_padding,
            missingness_color_column=missingness_color_column,
            point_opacity=point_opacity,
            jitter=jitter,
            missing_jitter=missing_jitter,
            jitter_seed=jitter_seed,
            xaxis_range=xaxis_range,
            yaxis_range=yaxis_range,
            **kwargs,
        )

    def venn(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        order: Literal["desc", "asc"] = "desc",
        value: Literal["count", "percent"] = "count",
        show_values: bool = True,
        missing_color: str = "#d62728",
        **kwargs,
    ) -> "_Venn":
        """Create a bar chart of the 7 exclusive missingness subsets for 3 columns."""
        from missingfcup.plots._venn import _Venn

        return _Venn(
            data=_md(self),
            selected_columns=selected_columns,
            order=order,
            value=value,
            show_values=show_values,
            missing_color=missing_color,
            **kwargs,
        )

    def upset(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        max_sets: int = 3,
        max_intersections: int = 20,
        min_intersection_size: int = 1,
        order: Literal["desc", "asc"] = "desc",
        show_values: bool = True,
        matrix_dot_size: int = 12,
        matrix_line_width: int = 3,
        excluded_dot_color: str = "#e0e0e0",
        highlight_columns: Optional[List[str]] = None,
        highlight_color: Optional[str] = None,
        **kwargs,
    ) -> "_Upset":
        """Create an UpSet plot of missingness intersections across columns."""
        from missingfcup.plots._upset import _Upset

        return _Upset(
            data=_md(self),
            selected_columns=selected_columns,
            max_sets=max_sets,
            max_intersections=max_intersections,
            min_intersection_size=min_intersection_size,
            order=order,
            show_values=show_values,
            matrix_dot_size=matrix_dot_size,
            matrix_line_width=matrix_line_width,
            excluded_dot_color=excluded_dot_color,
            highlight_columns=highlight_columns,
            highlight_color=highlight_color,
            **kwargs,
        )

    def parallel_coordinates(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        missingness_color_column: Optional[str] = None,
        max_columns: int = 0,
        line_opacity: float = 0.4,
        line_width: float = 1.0,
        missingness_only: bool = False,
        **kwargs,
    ) -> "_ParallelCoordinates":
        """
        Create a parallel coordinates plot (ggally style).

        Columns on x-axis, normalized values on y-axis [0, 1].
        Lines colored by missingness of ``missingness_color_column``:
        green = !NA, red = NA. ``max_columns`` caps how many axes are drawn
        (0 means no cap); useful when a wide dataset gives too many axes to read.
        """
        from missingfcup.plots._parallel_coordinates import _ParallelCoordinates

        return _ParallelCoordinates(
            data=_md(self),
            selected_columns=selected_columns,
            missingness_color_column=missingness_color_column,
            max_columns=max_columns,
            line_opacity=line_opacity,
            line_width=line_width,
            missingness_only=missingness_only,
            **kwargs,
        )

    def density(
        self,
        x: str,
        color_by: str,
        *,
        n_points: int = 300,
        fill_opacity: float = 0.4,
        **kwargs,
    ) -> "_Density":
        """
        Overlapping KDE density curves split by missingness of ``color_by``.

        Shows how the distribution of ``x`` differs between rows where
        ``color_by`` is present (!NA) versus missing (NA).
        Diverging distributions suggest MAR or MNAR; overlapping suggests MCAR.
        """
        from missingfcup.plots._density import _Density

        return _Density(
            data=_md(self),
            x=x,
            color_by=color_by,
            n_points=n_points,
            fill_opacity=fill_opacity,
            **kwargs,
        )

    def heatmap(
        self,
        *,
        kind: Literal["correlation", "predictive", "biserial"] = "correlation",
        selected_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.9,
        colorscale: str = "RdBu",
        show_values: bool = True,
        max_columns: int = 0,  # 0 = show all variables by default
        drop_constant_columns: Optional[bool] = None,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 1,
        show_colorbar: bool = True,
        show_upper_triangle: bool = False,
        nan_color: str = "#c7c7c7",
        text_font_size: int = 12,
        selected_value_columns: Optional[List[str]] = None,
        selected_missing_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> "_Plot":
        """
        Create a missingness association heatmap.

        ``kind="correlation"`` (default): missingness correlation (columns that
        tend to miss together). ``kind="predictive"``: present-vs-missing
        correlation (does observing one column predict missingness in another?).
        ``kind="biserial"``: point-biserial association between the observed
        values of one column and the missingness of another (a key MAR signal).

        ``selected_value_columns`` / ``selected_missing_columns`` apply to
        ``kind="biserial"`` only. ``drop_constant_columns`` defaults to False for
        correlation/predictive and True for biserial when left unset.
        """
        if kind not in ("correlation", "predictive", "biserial"):
            raise ValueError(
                f"kind must be 'correlation', 'predictive', or 'biserial', got {kind!r}"
            )
        if kind != "biserial" and (
            selected_value_columns is not None or selected_missing_columns is not None
        ):
            raise ValueError(
                "selected_value_columns / selected_missing_columns are only valid "
                "for kind='biserial'"
            )

        # Preserve each underlying plot's original drop_constant_columns default.
        if drop_constant_columns is None:
            drop_constant_columns = kind == "biserial"

        shared = dict(
            data=_md(self),
            selected_columns=selected_columns,
            ignore_high_missingness=ignore_high_missingness,
            high_missingness_threshold=high_missingness_threshold,
            colorscale=colorscale,
            show_values=show_values,
            max_columns=max_columns,
            drop_constant_columns=drop_constant_columns,
            order_by_missingness=order_by_missingness,
            order=order,
            value_round=value_round,
            show_colorbar=show_colorbar,
            show_upper_triangle=show_upper_triangle,
            nan_color=nan_color,
            text_font_size=text_font_size,
        )
        if kind == "correlation":
            from missingfcup.plots._heatmap_correlation import _HeatmapCorrelation

            return _HeatmapCorrelation(**shared, **kwargs)
        if kind == "predictive":
            from missingfcup.plots._heatmap_predictive import _HeatmapPredictive

            return _HeatmapPredictive(**shared, **kwargs)

        from missingfcup.plots._heatmap_biserial import _HeatmapBiserial

        return _HeatmapBiserial(
            **shared,
            selected_value_columns=selected_value_columns,
            selected_missing_columns=selected_missing_columns,
            **kwargs,
        )

    def rate(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.9,
        scale: Literal["fraction", "percentage"] = "fraction",
        colorscale: str = "Reds",
        show_values: bool = True,
        max_columns: int = 30,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 2,
        show_colorbar: bool = True,
        max_labels_with_values: int = 20,
        **kwargs,
    ) -> "_Rate":
        """Create a single-row heatmap of missing rates per column."""
        from missingfcup.plots._rate import _Rate

        return _Rate(
            data=_md(self),
            selected_columns=selected_columns,
            ignore_high_missingness=ignore_high_missingness,
            high_missingness_threshold=high_missingness_threshold,
            scale=scale,
            colorscale=colorscale,
            show_values=show_values,
            max_columns=max_columns,
            order_by_missingness=order_by_missingness,
            order=order,
            value_round=value_round,
            show_colorbar=show_colorbar,
            max_labels_with_values=max_labels_with_values,
            **kwargs,
        )

    def dendrogram(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.9,
        max_columns: int = 30,
        drop_constant_columns: bool = False,
        linkage_method: Literal[
            "single", "complete", "average", "weighted", "centroid", "median", "ward"
        ] = "average",
        use_abs_correlation: bool = False,
        line_width: int = 2,
        line_color: str = "#4C78A8",
        **kwargs,
    ) -> "_Dendrogram":
        """Create a dendrogram of missingness correlation between columns."""
        from missingfcup.plots._dendrogram import _Dendrogram

        return _Dendrogram(
            data=_md(self),
            selected_columns=selected_columns,
            ignore_high_missingness=ignore_high_missingness,
            high_missingness_threshold=high_missingness_threshold,
            max_columns=max_columns,
            drop_constant_columns=drop_constant_columns,
            linkage_method=linkage_method,
            use_abs_correlation=use_abs_correlation,
            line_width=line_width,
            line_color=line_color,
            **kwargs,
        )

    def boxplot(
        self,
        x: str,
        color_by: str,
        *,
        plot_type: Literal["box", "violin"] = "box",
        **kwargs,
    ) -> "_Boxplot":
        """
        Create a box (or violin) plot comparing the distribution of ``x``
        between rows where ``color_by`` is present vs. missing.

        Useful for diagnosing MAR and MNAR:
        * Different distributions: missingness of ``color_by`` may relate to values of ``x``
        * Similar distributions: consistent with MCAR

        Parameters
        ----------
        x : str
            Column whose value distribution to plot on the y-axis.
        color_by : str
            Column whose missingness splits the two groups (present vs. missing).
        plot_type : str
            ``"box"`` for box plots, ``"violin"`` for violin plots.

        Example
        -------
        md.boxplot(x="fare", color_by="age")
        answers "Do passengers with missing age tend to pay different fares?"
        """
        from missingfcup.plots._boxplot import _Boxplot

        return _Boxplot(
            data=_md(self),
            x=x,
            color_by=color_by,
            plot_type=plot_type,
            **kwargs,
        )
