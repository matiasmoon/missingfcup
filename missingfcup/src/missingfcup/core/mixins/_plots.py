from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Dict, Literal

if TYPE_CHECKING:
    from missingfcup.plots._barchart_missing_count import _MissingCountBarChart
    from missingfcup.plots._barchart_overall_missingness import _BarchartOverallMissingness
    from missingfcup.plots._heatmap import _Heatmap
    from missingfcup.plots._scatterplot import _ScatterPlot
    from missingfcup.plots._barchart_venn import _BarchartVenn
    from missingfcup.plots._barchart_intersection import _BarchartIntersection
    from missingfcup.plots._parallel_coordinates import _ParallelCoordinates
    from missingfcup.plots._heatmap_correlation import _HeatmapCorrelation
    from missingfcup.plots._heatmap_missing_rate_column import _ColumnMissingRateHeatmap
    from missingfcup.plots._dendrogram import _Dendrogram
    from missingfcup.plots._boxplot_missingness import _BoxplotMissingness
    from missingfcup.plots._density_missingness import _DensityMissingness
    from missingfcup.plots._heatmap_value_missingness import _HeatmapValueMissingness


class _MissingDataPlotMixin:
    """
    All plot factory methods for MissingData.

    Each method constructs and returns a plot object. Imports are lazy
    (inside the method body) to avoid circular imports between core and plots.
    """

    def barchart_missing_count(
        self,
        *,
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
    ) -> "_MissingCountBarChart":
        """Create a bar chart of missing (or present) counts per column."""
        from missingfcup.plots._barchart_missing_count import _MissingCountBarChart

        return _MissingCountBarChart(
            data=self,
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
            value=value,
            show_both=show_both,
            **kwargs,
        )

    def barchart_overall_missingness(self, **kwargs) -> "_BarchartOverallMissingness":
        """Create a bar chart showing total present vs missing cell counts across the dataset."""
        from missingfcup.plots._barchart_overall_missingness import _BarchartOverallMissingness

        return _BarchartOverallMissingness(data=self, **kwargs)

    def heatmap_missingness(
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
        max_label_length: int = 48,
        order_by_border_color: str = "#1f77b4",
        order_by_border_width: int = 2,
        **kwargs,
    ) -> "_Heatmap":
        """Create an interactive binary missingness heatmap."""
        from missingfcup.plots._heatmap import _Heatmap

        return _Heatmap(
            data=self,
            selected_columns=selected_columns,
            ignore_high_missingness=ignore_high_missingness,
            high_missingness_threshold=high_missingness_threshold,
            max_columns=max_columns,
            order_by=order_by,
            show_colorscale=show_colorscale,
            group_by_mode=group_by_mode,
            xgap=xgap,
            ygap=ygap,
            max_label_length=max_label_length,
            order_by_border_color=order_by_border_color,
            order_by_border_width=order_by_border_width,
            **kwargs,
        )

    def scatterplot_missingness(
        self,
        x: str,
        y: str,
        *,
        point_size: int = 8,
        axis_padding: float = 0.1,
        missingness_color_column: Optional[str] = None,
        missing_jitter: float = 0.5,
        xaxis_range: Optional[list] = None,
        yaxis_range: Optional[list] = None,
        **kwargs,
    ) -> "_ScatterPlot":
        """Create a scatter plot that keeps missing values visible via axis offsets."""
        from missingfcup.plots._scatterplot import _ScatterPlot

        return _ScatterPlot(
            data=self,
            x=x,
            y=y,
            point_size=point_size,
            axis_padding=axis_padding,
            missingness_color_column=missingness_color_column,
            missing_jitter=missing_jitter,
            xaxis_range=xaxis_range,
            yaxis_range=yaxis_range,
            **kwargs,
        )

    def barchart_venn(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        sort_order: Literal["desc", "asc"] = "desc",
        value: Literal["count", "percent"] = "count",
        show_values: bool = True,
        max_label_length: int = 48,
        missing_color: str = "#d62728",
        **kwargs,
    ) -> "_BarchartVenn":
        """Create a bar chart of the 7 exclusive missingness subsets for 3 columns."""
        from missingfcup.plots._barchart_venn import _BarchartVenn

        return _BarchartVenn(
            data=self,
            selected_columns=selected_columns,
            sort_order=sort_order,
            value=value,
            show_values=show_values,
            max_label_length=max_label_length,
            missing_color=missing_color,
            **kwargs,
        )

    def barchart_intersection(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        max_sets: int = 3,
        max_intersections: int = 20,
        min_intersection_size: int = 1,
        sort_order: Literal["desc", "asc"] = "desc",
        show_values: bool = True,
        matrix_dot_size: int = 12,
        matrix_line_width: int = 3,
        excluded_dot_color: str = "#e0e0e0",
        highlight_columns: Optional[List[str]] = None,
        highlight_color: Optional[str] = None,
        **kwargs,
    ) -> "_BarchartIntersection":
        """Create a bar chart of missingness intersections across columns."""
        from missingfcup.plots._barchart_intersection import _BarchartIntersection

        return _BarchartIntersection(
            data=self,
            selected_columns=selected_columns,
            max_sets=max_sets,
            max_intersections=max_intersections,
            min_intersection_size=min_intersection_size,
            sort_order=sort_order,
            show_values=show_values,
            matrix_dot_size=matrix_dot_size,
            matrix_line_width=matrix_line_width,
            excluded_dot_color=excluded_dot_color,
            highlight_columns=highlight_columns,
            highlight_color=highlight_color,
            **kwargs,
        )

    def parallel_coordinates_missingness(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        missingness_color_column: Optional[str] = None,
        line_opacity: float = 0.4,
        line_width: float = 1.0,
        missingness_only: bool = False,
        **kwargs,
    ) -> "_ParallelCoordinates":
        """
        Create a parallel coordinates plot (ggally style).

        Columns on x-axis, normalized values on y-axis [0, 1].
        Lines coloured by missingness of ``missingness_color_column``:
        green = !NA, red = NA.
        """
        from missingfcup.plots._parallel_coordinates import _ParallelCoordinates

        return _ParallelCoordinates(
            data=self,
            selected_columns=selected_columns,
            missingness_color_column=missingness_color_column,
            line_opacity=line_opacity,
            line_width=line_width,
            missingness_only=missingness_only,
            **kwargs,
        )

    def density_missingness(
        self,
        x: str,
        color_by: str,
        *,
        n_points: int = 300,
        fill_opacity: float = 0.4,
        **kwargs,
    ) -> "_DensityMissingness":
        """
        Overlapping KDE density curves split by missingness of ``color_by``.

        Shows how the distribution of ``x`` differs between rows where
        ``color_by`` is present (!NA) versus missing (NA).
        Diverging distributions suggest MAR or MNAR; overlapping suggests MCAR.
        """
        from missingfcup.plots._density_missingness import _DensityMissingness

        return _DensityMissingness(
            data=self,
            x=x,
            color_by=color_by,
            n_points=n_points,
            fill_opacity=fill_opacity,
            **kwargs,
        )

    def heatmap_correlation_missing_missing(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        colorscale: str = "RdBu",
        show_values: bool = True,
        max_columns: int = 0,  # 0 = show all variables by default
        drop_constant_columns: bool = False,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 1,
        show_colorbar: bool = True,
        show_upper_triangle: bool = False,
        nan_color: str = "#c7c7c7",
        **kwargs,
    ) -> "_HeatmapCorrelation":
        """Create a heatmap of missingness correlation — columns that tend to miss together."""
        from missingfcup.plots._heatmap_correlation import _HeatmapCorrelation

        return _HeatmapCorrelation(
            data=self,
            mode="missing_missing",
            selected_columns=selected_columns,
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
            **kwargs,
        )

    def heatmap_correlation_present_present(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        colorscale: str = "RdBu",
        show_values: bool = True,
        max_columns: int = 0,  # 0 = show all variables by default
        drop_constant_columns: bool = False,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 1,
        show_colorbar: bool = True,
        show_upper_triangle: bool = False,
        nan_color: str = "#c7c7c7",
        **kwargs,
    ) -> "_HeatmapCorrelation":
        """Create a heatmap of presence correlation — columns that tend to be observed together."""
        from missingfcup.plots._heatmap_correlation import _HeatmapCorrelation

        return _HeatmapCorrelation(
            data=self,
            mode="present_present",
            selected_columns=selected_columns,
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
            **kwargs,
        )

    def heatmap_correlation_present_missing(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        colorscale: str = "RdBu",
        show_values: bool = True,
        max_columns: int = 0,  # 0 = show all variables by default
        drop_constant_columns: bool = False,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 1,
        show_colorbar: bool = True,
        show_upper_triangle: bool = False,
        nan_color: str = "#c7c7c7",
        **kwargs,
    ) -> "_HeatmapCorrelation":
        """Create a heatmap of present-vs-missing correlation — does observing one column predict missingness in another?"""
        from missingfcup.plots._heatmap_correlation import _HeatmapCorrelation

        return _HeatmapCorrelation(
            data=self,
            mode="present_missing",
            selected_columns=selected_columns,
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
            **kwargs,
        )

    def heatmap_missing_rate(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        scale: Literal["fraction", "percentage"] = "fraction",
        colorscale: str = "Reds",
        show_values: bool = True,
        max_columns: int = 30,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 2,
        show_colorbar: bool = True,
        max_label_length: int = 48,
        max_labels_with_values: int = 20,
        **kwargs,
    ) -> "_ColumnMissingRateHeatmap":
        """Create a single-row heatmap of missing rates per column."""
        from missingfcup.plots._heatmap_missing_rate_column import _ColumnMissingRateHeatmap

        return _ColumnMissingRateHeatmap(
            data=self,
            selected_columns=selected_columns,
            scale=scale,
            colorscale=colorscale,
            show_values=show_values,
            max_columns=max_columns,
            order_by_missingness=order_by_missingness,
            order=order,
            value_round=value_round,
            show_colorbar=show_colorbar,
            max_label_length=max_label_length,
            max_labels_with_values=max_labels_with_values,
            **kwargs,
        )

    def dendrogram_missingness(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
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
            data=self,
            selected_columns=selected_columns,
            max_columns=max_columns,
            drop_constant_columns=drop_constant_columns,
            linkage_method=linkage_method,
            use_abs_correlation=use_abs_correlation,
            line_width=line_width,
            line_color=line_color,
            **kwargs,
        )

    def heatmap_value_missingness_correlation(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        selected_value_columns: Optional[List[str]] = None,
        selected_missing_columns: Optional[List[str]] = None,
        colorscale: str = "RdBu",
        show_values: bool = True,
        max_columns: int = 0,
        drop_constant_columns: bool = True,
        order_by_missingness: bool = True,
        order: Literal["desc", "asc"] = "desc",
        value_round: int = 1,
        show_colorbar: bool = True,
        show_upper_triangle: bool = False,
        nan_color: str = "#c7c7c7",
        **kwargs,
    ) -> "_HeatmapValueMissingness":
        """
        Create a heatmap of value-missingness associations.

        Each cell [i, j] shows the point-biserial correlation between the observed
        values of column i and the missingness indicator of column j. A non-zero
        value means the distribution of i differs between rows where j is present
        vs. missing — a key signal for MAR diagnosis.

        Parameters
        ----------
        selected_columns : list[str], optional
            Shorthand to restrict both axes to the same set of columns.
        selected_value_columns : list[str], optional
            Columns to use as value predictors (y-axis). Overrides selected_columns.
        selected_missing_columns : list[str], optional
            Columns whose missingness to predict (x-axis). Overrides selected_columns.
        show_upper_triangle : bool, optional
            If False (default), the upper triangle is hidden for square matrices.
        """
        from missingfcup.plots._heatmap_value_missingness import _HeatmapValueMissingness

        return _HeatmapValueMissingness(
            data=self,
            selected_columns=selected_columns,
            selected_value_columns=selected_value_columns,
            selected_missing_columns=selected_missing_columns,
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
            **kwargs,
        )

    def boxplot_missingness(
        self,
        x: str,
        color_by: str,
        *,
        plot_type: Literal["box", "violin"] = "box",
        **kwargs,
    ) -> "_BoxplotMissingness":
        """
        Create a box (or violin) plot comparing the distribution of ``x``
        between rows where ``color_by`` is present vs. missing.

        Useful for diagnosing MAR and MNAR:
        - Different distributions → missingness of ``color_by`` may relate to values of ``x``
        - Similar distributions → consistent with MCAR

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
        md.boxplot_missingness(x="fare", color_by="age")
        → "Do passengers with missing age tend to pay different fares?"
        """
        from missingfcup.plots._boxplot_missingness import _BoxplotMissingness

        return _BoxplotMissingness(
            data=self,
            x=x,
            color_by=color_by,
            plot_type=plot_type,
            **kwargs,
        )
