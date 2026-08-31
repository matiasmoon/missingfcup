from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, Sequence, Union, cast

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

# The heatmap kinds that read a column's values rather than its missingness mask.
# They share an asymmetric shape, so the options and the guards that follow from it
# apply to both and to neither of the others.
_VALUE_KINDS = ("direction", "dependence")


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

    Naming convention
    -----------------
    These signatures are the package's public surface, so a name has to say what a
    parameter does before anyone reads the docstring. New parameters follow the
    shapes already in use:

    ==================  ======  ================================================
    Shape               Type    Means
    ==================  ======  ================================================
    ``show_*``          bool    changes what is **drawn**, never the numbers
    ``drop_*``          bool    changes which **data** is included
    ``use_*``           bool    changes **how something is computed**
    ``selected_*``      list    the caller names the columns to use
    ``sort_*``          mixed   ordering
    ``max_*``           int     a cap; 0 means no cap
    ``*_threshold``     float   a cutoff compared against a rate
    ``*_color``         str     a colour
    ``*_range``         list    an explicit ``[min, max]``
    ``kind``            choice  which variant of this plot to **compute**
    ``shape``           choice  how one computed result is **drawn**
    ``measure``         choice  which quantity to show: count, fraction, percent
    bare noun           any     the one obvious property (``title``, ``width``)
    ==================  ======  ================================================

    The three boolean prefixes carry the load: a reader can tell from the prefix
    alone whether a flag is cosmetic (``show_``) or whether it changes the numbers
    (``drop_``, ``use_``), which matters more here than in a general plotting
    library. Only one polarity is used -- there is no ``include_*`` to sit opposite
    ``drop_*``, because two spellings of one switch is how they end up disagreeing.

    ``ascending`` is a deliberate exception: bare, no prefix, taken verbatim from
    ``DataFrame.sort_values``. Matching pandas is worth more than matching this
    table, and it pairs with ``sort_by`` the way a user already expects.

    ``kind`` and ``shape`` split what used to be one word. ``kind`` changes the
    result: ``heatmap(kind=)`` picks the statistic and the data it reads, and
    ``parallel_coordinates(kind=)`` picks between the values and the missingness
    mask. ``shape`` cannot change any number -- ``boxplot(shape="violin")`` draws
    the identical data as a violin instead of a box. Spelling both ``kind`` taught
    a reader to expect a different computation from a parameter that only ever
    changed the rendering.

    A parameter that cannot apply to a given call raises rather than being ignored.
    ``show_both`` on a rate measure, ``show_upper_triangle`` on the asymmetric
    heatmap, and ``sort_categories`` without a sort column all refuse the call,
    because silently dropping an argument looks exactly like honouring it.
    """

    def bar(
        self,
        *,
        measure: Literal["count", "fraction", "percentage"] = "count",
        selected_columns: Optional[List[str]] = None,
        high_missingness_threshold: Optional[float] = None,
        max_columns: int = 0,
        sort_by: Optional[Literal["missingness", "alphabetical"]] = None,
        ascending: bool = False,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        show_values: bool = True,
        show_both: bool = False,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        palette: str = "default",
        missing_color: Optional[str] = None,
        present_color: Optional[str] = None,
        show_legend: bool = True,
        max_label_length: int = 48,
    ) -> "_Plot":
        """
        Create a per-column bar chart of missingness.

        ``measure="count"`` (default) plots the missing count per column, and
        ``show_both`` stacks the present count on top of it. ``"fraction"`` and
        ``"percentage"`` plot the missing rate on a 0-1 or 0-100 scale.

        Parameters
        ----------
        measure : {"count", "fraction", "percentage"}, default "count"
            An absolute count, or the same number as a share of the dataset on a
            0-1 or a 0-100 scale.
        selected_columns : list of str, optional
            Restrict the chart to these columns. Applied after the
            ``high_missingness_threshold`` filter, so a column dropped there stays dropped.
        high_missingness_threshold : float, optional
            Drop columns whose missing rate reaches this value, before anything
            else. ``None``, the default, keeps every column: the emptiest ones are
            usually what a missing-data plot is for.
        max_columns : int, default 0
            Hard cap on how many columns are drawn, applied last. The first
            ``max_columns`` surviving columns are kept. 0 means no cap.
        sort_by : {"missingness", "alphabetical"}, optional
            What to order the columns on. ``None`` keeps the DataFrame's own
            order. Pairs with ``ascending``, as in pandas.
        ascending : bool, default False
            Sort direction. The default puts the emptiest columns first.
        orientation : {"vertical", "horizontal"}, default "vertical"
            Whether bars run up from the x-axis or across from the y-axis.
        show_values : bool, default True
            Draw the numeric value on each bar.
        show_both : bool, default False
            Stack present on top of missing so each bar's height is the row count.
            ``measure="count"`` only; raises for the rate measures, where a stacked
            bar would always reach 1.0 and say nothing.
        title : str, optional
            Title drawn above the figure. Also names the PNG that the toolbar's
            download button writes.
        width : int, default 900
            Figure width in pixels. Capped at 2000.
        height : int, default 520
            Figure height in pixels. Capped at 1000.
        background_color : str, optional
            Paper and plot background colour. ``None`` keeps the plotly default.
        text_color : str, optional
            Font colour for the figure. ``None`` keeps the plotly default.
        palette : {"default", "safe", "grayscale"}, default "default"
            Named colour pair for the present/missing distinction. ``"default"`` is red
            against green, which is the worst case for red-green colour vision
            deficiency and measures 1.48:1 in normal vision. ``"safe"`` and
            ``"grayscale"`` both clear the 3:1 WCAG floor for non-text content under
            simulated deuteranopia, protanopia and tritanopia; ``"grayscale"`` also
            survives monochrome printing.
        missing_color : str, optional
            Colour used to draw missing values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        present_color : str, optional
            Colour used to draw present values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        show_legend : bool, default True
            Whether to draw the legend.
        max_label_length : int, default 48
            Axis labels longer than this are truncated with an ellipsis, then
            de-duplicated if truncation made two labels identical. 0 falls back to a
            budget derived from ``width``.

        Returns
        -------
        _Plot
            A ``_BarCount`` or ``_BarRate``, depending on ``measure``.
        """
        shared = dict(
            data=_md(self),
            selected_columns=selected_columns,
            high_missingness_threshold=high_missingness_threshold,
            max_columns=max_columns,
            sort_by=sort_by,
            ascending=ascending,
            orientation=orientation,
            show_values=show_values,
            title=title,
            width=width,
            height=height,
            background_color=background_color,
            text_color=text_color,
            palette=palette,
            missing_color=missing_color,
            present_color=present_color,
            show_legend=show_legend,
            max_label_length=max_label_length,
        )
        if measure == "count":
            from missingfcup.plots._bar_count import _BarCount

            return _BarCount(**shared, show_both=show_both)
        if measure in ("fraction", "percentage"):
            if show_both:
                # Dropping it quietly would draw a plain rate bar, looking like it
                # had been honoured.
                raise ValueError(
                    f"show_both applies to measure='count' only, not "
                    f"measure={measure!r}: missing and present rates always sum to 1, "
                    f"so every stacked bar would be the same height."
                )
            from missingfcup.plots._bar_rate import _BarRate

            return _BarRate(**shared, measure=measure)
        raise ValueError(f"measure must be 'count', 'fraction' or 'percentage', got {measure!r}")

    def totals(
        self,
        *,
        show_values: bool = True,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        palette: str = "default",
        missing_color: Optional[str] = None,
        present_color: Optional[str] = None,
        max_label_length: int = 48,
    ) -> "_Totals":
        """
        Create a bar chart showing total present vs missing cell counts across the dataset.

        Two bars for the whole DataFrame, with no per-column breakdown.

        Parameters
        ----------
        show_values : bool, default True
            Write the count and its share of all cells above each bar.
        title : str, optional
            Title drawn above the figure. Also names the PNG that the toolbar's
            download button writes.
        width : int, default 900
            Figure width in pixels. Capped at 2000.
        height : int, default 520
            Figure height in pixels. Capped at 1000.
        background_color : str, optional
            Paper and plot background colour. ``None`` keeps the plotly default.
        text_color : str, optional
            Font colour for the figure. ``None`` keeps the plotly default.
        palette : {"default", "safe", "grayscale"}, default "default"
            Named colour pair for the present/missing distinction. ``"default"`` is red
            against green, which is the worst case for red-green colour vision
            deficiency and measures 1.48:1 in normal vision. ``"safe"`` and
            ``"grayscale"`` both clear the 3:1 WCAG floor for non-text content under
            simulated deuteranopia, protanopia and tritanopia; ``"grayscale"`` also
            survives monochrome printing.
        missing_color : str, optional
            Colour used to draw missing values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        present_color : str, optional
            Colour used to draw present values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        max_label_length : int, default 48
            Axis labels longer than this are truncated with an ellipsis, then
            de-duplicated if truncation made two labels identical. 0 falls back to a
            budget derived from ``width``.

        Returns
        -------
        _Totals
        """
        from missingfcup.plots._totals import _Totals

        return _Totals(
            data=_md(self),
            show_values=show_values,
            title=title,
            width=width,
            height=height,
            background_color=background_color,
            text_color=text_color,
            palette=palette,
            missing_color=missing_color,
            present_color=present_color,
            max_label_length=max_label_length,
        )

    def matrix(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        high_missingness_threshold: Optional[float] = None,
        max_columns: int = 0,
        sort_by: Optional[Union[Literal["missingness", "alphabetical"], str]] = None,
        ascending: bool = False,
        sort_categories: Optional[Sequence] = None,
        show_legend: bool = True,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        palette: str = "default",
        missing_color: Optional[str] = None,
        present_color: Optional[str] = None,
        max_label_length: int = 48,
    ) -> "_Matrix":
        """
        Create an interactive binary missingness matrix (nullity matrix).

        One row per observation and one column per variable, with each cell coloured
        by whether that value is present or missing.

        Parameters
        ----------
        selected_columns : list of str, optional
            Restrict the plot to these columns. Applied after the
            ``high_missingness_threshold`` filter, so a column dropped there stays
            dropped. Raises ValueError if nothing survives.
        high_missingness_threshold : float, optional
            Drop columns whose missing rate reaches this value, before anything
            else. ``None``, the default, keeps every column: the emptiest ones are
            usually what a missing-data plot is for.
        max_columns : int, default 0
            Hard cap on how many columns are drawn. 0 means no cap.
        sort_by : {"missingness", "alphabetical"} or str, optional
            What to sort on. The two keywords order the *columns*; the name of a
            column in the DataFrame instead orders the *rows* by that column's
            values, and pins it to the left edge so the ordering is readable.
            ``None`` keeps the DataFrame's own order. Pairs with ``ascending``,
            as in pandas.

            Rows are ordered with ``DataFrame.sort_values``, so the column's dtype
            decides the order: numerically for numbers, lexicographically for
            strings, and by declared category order for a categorical. To put a
            categorical in a meaningful order rather than an alphabetical one, say
            so on the column itself and every plot follows::

                df["size"] = pd.Categorical(df["size"], categories=["S", "M", "L"])
                md.matrix(sort_by="size")  # draws S, then M, then L

            Missing values sort last whichever direction ``ascending`` gives, since
            reversing the sort should not move the rows under study off the far end
            of the figure.
        ascending : bool, default False
            Sort direction. The default puts the emptiest columns first.
        sort_categories : sequence, optional
            The exact order to draw the values of the column ``sort_by`` names, for
            when their natural order means nothing. A nominal categorical has no
            inherent first or last, so alphabetical is an arbitrary choice, and this
            is how the caller makes it deliberately::

                md.matrix(sort_by="country",
                          sort_categories=["Portugal", "France", "Spain"])

            The sequence is drawn exactly as written, so ``ascending`` does not
            apply to it: the sequence already says which value is first and which is
            last. Reverse the sequence to reverse the plot. Values not named come
            after the ones that are, and missing values come last. Raises ValueError
            if ``sort_by`` names no column, or if none of the values given appear in
            that column.
        show_legend : bool, default True
            Whether to draw the colour key beside the plot.
        title : str, optional
            Title drawn above the figure. Also names the PNG that the toolbar's
            download button writes.
        width : int, default 900
            Figure width in pixels. Capped at 2000.
        height : int, default 520
            Figure height in pixels. Capped at 1000.
        background_color : str, optional
            Paper and plot background colour. ``None`` keeps the plotly default.
        text_color : str, optional
            Font colour for the figure. ``None`` keeps the plotly default.
        palette : {"default", "safe", "grayscale"}, default "default"
            Named colour pair for the present/missing distinction. ``"default"`` is red
            against green, which is the worst case for red-green colour vision
            deficiency and measures 1.48:1 in normal vision. ``"safe"`` and
            ``"grayscale"`` both clear the 3:1 WCAG floor for non-text content under
            simulated deuteranopia, protanopia and tritanopia; ``"grayscale"`` also
            survives monochrome printing.
        missing_color : str, optional
            Colour used to draw missing values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        present_color : str, optional
            Colour used to draw present values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        max_label_length : int, default 48
            Axis labels longer than this are truncated with an ellipsis, then
            de-duplicated if truncation made two labels identical. 0 falls back to a
            budget derived from ``width``.

        Returns
        -------
        _Matrix
        """
        from missingfcup.plots._matrix import _Matrix

        return _Matrix(
            data=_md(self),
            selected_columns=selected_columns,
            high_missingness_threshold=high_missingness_threshold,
            max_columns=max_columns,
            sort_by=sort_by,
            ascending=ascending,
            sort_categories=sort_categories,
            show_legend=show_legend,
            title=title,
            width=width,
            height=height,
            background_color=background_color,
            text_color=text_color,
            palette=palette,
            missing_color=missing_color,
            present_color=present_color,
            max_label_length=max_label_length,
        )

    def scatterplot(
        self,
        x: str,
        y: str,
        *,
        axis_padding: float = 0.1,
        missing_column: Optional[str] = None,
        jitter: float = 0.02,
        jitter_seed: int = 42,
        xaxis_range: Optional[list] = None,
        yaxis_range: Optional[list] = None,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        palette: str = "default",
        missing_color: Optional[str] = None,
        present_color: Optional[str] = None,
        show_legend: bool = True,
        max_label_length: int = 48,
    ) -> "_Scatterplot":
        """
        Create a scatter plot that keeps missing values visible via axis offsets.

        A normal scatter drops any row where either axis is missing. Here those rows
        are offset outside the data range with a distinct marker, so the values that
        cannot be plotted stay visible and countable.

        Parameters
        ----------
        x : str
            Column drawn on the x-axis.
        y : str
            Column drawn on the y-axis.
        axis_padding : float, default 0.1
            Fraction of the data span added as padding at each end of both axes. This
            widens the frame only; the offset markers for missing values sit a fixed
            tenth of the observed span outside the data either way, so ``0.0`` still
            draws them, flush against the edge rather than clear of it.
        missing_column : str, optional
            Colour points by whether this column is missing in that row, instead of by
            the missingness of ``x`` and ``y``. Raises ValueError if the column is not
            in the DataFrame.
        jitter : float, default 0.02
            Gaussian noise added to every plotted point, as a fraction of each axis
            span, so that coincident values separate instead of stacking into one
            mark. It applies to the offset markers for missing values too, which
            share a single coordinate and would otherwise draw as one point however
            many rows they stand for; theirs is capped so the band cannot spread far
            enough to overlap the observed range. 0 disables it, giving exact
            positions at the cost of hiding density.
        jitter_seed : int, default 42
            Seed for the jitter, so a figure is reproducible.
        xaxis_range : list, optional
            Explicit ``[min, max]`` for the x-axis, overriding the computed range.
        yaxis_range : list, optional
            Explicit ``[min, max]`` for the y-axis.
        title : str, optional
            Title drawn above the figure. Also names the PNG that the toolbar's
            download button writes.
        width : int, default 900
            Figure width in pixels. Capped at 2000.
        height : int, default 520
            Figure height in pixels. Capped at 1000.
        background_color : str, optional
            Paper and plot background colour. ``None`` keeps the plotly default.
        text_color : str, optional
            Font colour for the figure. ``None`` keeps the plotly default.
        palette : {"default", "safe", "grayscale"}, default "default"
            Named colour pair for the present/missing distinction. ``"default"`` is red
            against green, which is the worst case for red-green colour vision
            deficiency and measures 1.48:1 in normal vision. ``"safe"`` and
            ``"grayscale"`` both clear the 3:1 WCAG floor for non-text content under
            simulated deuteranopia, protanopia and tritanopia; ``"grayscale"`` also
            survives monochrome printing.
        missing_color : str, optional
            Colour used to draw missing values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        present_color : str, optional
            Colour used to draw present values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        show_legend : bool, default True
            Whether to draw the legend.
        max_label_length : int, default 48
            Axis labels longer than this are truncated with an ellipsis, then
            de-duplicated if truncation made two labels identical. 0 falls back to a
            budget derived from ``width``.

        Returns
        -------
        _Scatterplot
        """
        from missingfcup.plots._scatterplot import _Scatterplot

        return _Scatterplot(
            data=_md(self),
            x=x,
            y=y,
            axis_padding=axis_padding,
            missing_column=missing_column,
            jitter=jitter,
            jitter_seed=jitter_seed,
            xaxis_range=xaxis_range,
            yaxis_range=yaxis_range,
            title=title,
            width=width,
            height=height,
            background_color=background_color,
            text_color=text_color,
            palette=palette,
            missing_color=missing_color,
            present_color=present_color,
            show_legend=show_legend,
            max_label_length=max_label_length,
        )

    def venn(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        sort_by: Optional[Literal["size"]] = "size",
        ascending: bool = False,
        measure: Literal["count", "fraction", "percentage"] = "count",
        show_values: bool = True,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        palette: str = "default",
        missing_color: Optional[str] = None,
        present_color: Optional[str] = None,
        show_legend: bool = True,
        max_label_length: int = 48,
    ) -> "_Venn":
        """
        Create a bar chart of the 7 exclusive missingness subsets for 3 columns.

        Each bar is one region of a three-set Venn diagram: rows missing only A, only
        B, both A and B, and so on.

        Parameters
        ----------
        selected_columns : list of str
            The three columns to compare. Required, and it must name exactly three:
            a three-set Venn has 7 exclusive regions and no other number of columns
            defines the plot. Use ``upset()`` for a different number.
        sort_by : {"size"}, default "size"
            What to order the regions on. ``"size"`` orders them by how many rows
            they cover; ``None`` keeps the order the regions are enumerated in.
            Pairs with ``ascending``, as in pandas.
        ascending : bool, default False
            Sort direction. The default puts the largest regions first.
        measure : {"count", "fraction", "percentage"}, default "count"
            An absolute count, or the same number as a share of the dataset on a
            0-1 or a 0-100 scale.
        show_values : bool, default True
            Draw the numeric value on each bar.
        title : str, optional
            Title drawn above the figure. Also names the PNG that the toolbar's
            download button writes.
        width : int, default 900
            Figure width in pixels. Capped at 2000.
        height : int, default 520
            Figure height in pixels. Capped at 1000.
        background_color : str, optional
            Paper and plot background colour. ``None`` keeps the plotly default.
        text_color : str, optional
            Font colour for the figure. ``None`` keeps the plotly default.
        palette : {"default", "safe", "grayscale"}, default "default"
            Named colour pair for the present/missing distinction. ``"default"`` is red
            against green, which is the worst case for red-green colour vision
            deficiency and measures 1.48:1 in normal vision. ``"safe"`` and
            ``"grayscale"`` both clear the 3:1 WCAG floor for non-text content under
            simulated deuteranopia, protanopia and tritanopia; ``"grayscale"`` also
            survives monochrome printing.
        missing_color : str, optional
            Colour used to draw missing values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        present_color : str, optional
            Colour used to draw present values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        show_legend : bool, default True
            Whether to draw the legend.
        max_label_length : int, default 48
            Axis labels longer than this are truncated with an ellipsis, then
            de-duplicated if truncation made two labels identical. 0 falls back to a
            budget derived from ``width``.

        Returns
        -------
        _Venn
        """
        from missingfcup.plots._venn import _Venn

        return _Venn(
            data=_md(self),
            selected_columns=selected_columns,
            sort_by=sort_by,
            ascending=ascending,
            measure=measure,
            show_values=show_values,
            title=title,
            width=width,
            height=height,
            background_color=background_color,
            text_color=text_color,
            palette=palette,
            missing_color=missing_color,
            present_color=present_color,
            show_legend=show_legend,
            max_label_length=max_label_length,
        )

    def upset(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        sort_by: Optional[Literal["size"]] = "size",
        ascending: bool = False,
        measure: Literal["count", "fraction", "percentage"] = "count",
        show_values: bool = True,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        palette: str = "default",
        missing_color: Optional[str] = None,
        present_color: Optional[str] = None,
        max_label_length: int = 48,
    ) -> "_Upset":
        """
        Create an UpSet plot of missingness intersections across columns.

        The Venn approach stops working past three columns; UpSet scales. Bars give the
        size of each intersection and the dot matrix underneath says which columns that
        intersection covers.

        At most 20 intersection bars are drawn, largest first. A selection producing
        more warns rather than truncating quietly, since a partial view that looks
        complete is worse than a smaller one.

        Parameters
        ----------
        selected_columns : list of str
            Columns to compare, all of which are drawn. Required: picking them here
            would mean the figure answered a different question from the one asked.
        sort_by : {"size"}, default "size"
            What to order the intersections on. ``"size"`` orders them by how many
            rows they cover; ``None`` keeps the order the intersections are
            enumerated in. Pairs with ``ascending``, as in pandas.
        ascending : bool, default False
            Sort direction. The default puts the largest intersections first.
        measure : {"count", "fraction", "percentage"}, default "count"
            An absolute number of rows, or the same number as a share of the dataset
            on a 0-1 or a 0-100 scale, exactly as on ``venn()``. It scales both bar
            panels, so intersection sizes and set sizes stay on one scale.
        show_values : bool, default True
            Draw the size on each intersection bar.
        title : str, optional
            Title drawn above the figure. Also names the PNG that the toolbar's
            download button writes.
        width : int, default 900
            Figure width in pixels. Capped at 2000.
        height : int, default 520
            Figure height in pixels. Capped at 1000.
        background_color : str, optional
            Paper and plot background colour. ``None`` keeps the plotly default.
        text_color : str, optional
            Font colour for the figure. ``None`` keeps the plotly default.
        palette : {"default", "safe", "grayscale"}, default "default"
            Named colour pair for the present/missing distinction. ``"default"`` is red
            against green, which is the worst case for red-green colour vision
            deficiency and measures 1.48:1 in normal vision. ``"safe"`` and
            ``"grayscale"`` both clear the 3:1 WCAG floor for non-text content under
            simulated deuteranopia, protanopia and tritanopia; ``"grayscale"`` also
            survives monochrome printing.
        missing_color : str, optional
            Colour used to draw missing values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        present_color : str, optional
            Colour used to draw present values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        max_label_length : int, default 48
            Axis labels longer than this are truncated with an ellipsis, then
            de-duplicated if truncation made two labels identical. 0 falls back to a
            budget derived from ``width``.

        Returns
        -------
        _Upset
        """
        from missingfcup.plots._upset import _Upset

        return _Upset(
            data=_md(self),
            selected_columns=selected_columns,
            sort_by=sort_by,
            ascending=ascending,
            measure=measure,
            show_values=show_values,
            title=title,
            width=width,
            height=height,
            background_color=background_color,
            text_color=text_color,
            palette=palette,
            missing_color=missing_color,
            present_color=present_color,
            max_label_length=max_label_length,
        )

    def parallel_coordinates(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        missing_column: Optional[str] = None,
        high_missingness_threshold: Optional[float] = None,
        max_columns: int = 0,
        sort_by: Optional[Literal["missingness", "alphabetical"]] = None,
        ascending: bool = False,
        kind: Literal["values", "missingness"] = "values",
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        palette: str = "default",
        missing_color: Optional[str] = None,
        present_color: Optional[str] = None,
        show_legend: bool = True,
        max_label_length: int = 48,
    ) -> "_ParallelCoordinates":
        """
        Create a parallel coordinates plot (ggally style).

        Columns on x-axis, normalized values on y-axis [0, 1].
        Lines colored by missingness of ``missing_column``:
        green = !NA, red = NA. ``max_columns`` caps how many axes are drawn
        (0 means no cap); useful when a wide dataset gives too many axes to read.

        Parameters
        ----------
        selected_columns : list of str, optional
            Columns to draw as axes. When omitted, every column is used.
        missing_column : str, optional
            Colour each line by whether this column is missing in that row, drawing
            two series labelled ``"NA-<column>"`` and ``"!NA-<column>"``. When
            omitted every line is drawn the same colour and no legend appears,
            since a single unnamed series has nothing to distinguish.
        high_missingness_threshold : float, optional
            Drop columns whose missing rate reaches this value, before anything
            else. ``None``, the default, keeps every column.
        max_columns : int, default 0
            Keep only the first this many columns. 0 means no cap.
        sort_by : {"missingness", "alphabetical"}, optional
            What to order the axes on. ``None`` keeps the DataFrame's own order.
            Pairs with ``ascending``, as in pandas. Axis order carries more weight
            here than on other plots: a relationship between two columns is visible
            only where their axes are adjacent, so ordering by missingness puts the
            columns most likely to share gaps next to each other.
        ascending : bool, default False
            Sort direction. The default puts the emptiest columns first.
        kind : {"values", "missingness"}, default "values"
            What each axis encodes. ``"values"`` normalises the column onto a shared
            0-1 scale, so a line traces a row's actual measurements. ``"missingness"``
            draws every column as binary present/missing instead, which is the escape
            hatch for non-numeric columns that cannot be normalised at all.
        title : str, optional
            Title drawn above the figure. Also names the PNG that the toolbar's
            download button writes.
        width : int, default 900
            Figure width in pixels. Capped at 2000.
        height : int, default 520
            Figure height in pixels. Capped at 1000.
        background_color : str, optional
            Paper and plot background colour. ``None`` keeps the plotly default.
        text_color : str, optional
            Font colour for the figure. ``None`` keeps the plotly default.
        palette : {"default", "safe", "grayscale"}, default "default"
            Named colour pair for the present/missing distinction. ``"default"`` is red
            against green, which is the worst case for red-green colour vision
            deficiency and measures 1.48:1 in normal vision. ``"safe"`` and
            ``"grayscale"`` both clear the 3:1 WCAG floor for non-text content under
            simulated deuteranopia, protanopia and tritanopia; ``"grayscale"`` also
            survives monochrome printing.
        missing_color : str, optional
            Colour used to draw missing values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        present_color : str, optional
            Colour used to draw present values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        show_legend : bool, default True
            Whether to draw the legend.
        max_label_length : int, default 48
            Axis labels longer than this are truncated with an ellipsis, then
            de-duplicated if truncation made two labels identical. 0 falls back to a
            budget derived from ``width``.

        Returns
        -------
        _ParallelCoordinates
        """
        from missingfcup.plots._parallel_coordinates import _ParallelCoordinates

        return _ParallelCoordinates(
            data=_md(self),
            selected_columns=selected_columns,
            missing_column=missing_column,
            high_missingness_threshold=high_missingness_threshold,
            max_columns=max_columns,
            sort_by=sort_by,
            ascending=ascending,
            kind=kind,
            title=title,
            width=width,
            height=height,
            background_color=background_color,
            text_color=text_color,
            palette=palette,
            missing_color=missing_color,
            present_color=present_color,
            show_legend=show_legend,
            max_label_length=max_label_length,
        )

    def density(
        self,
        column: str,
        missing_column: str,
        *,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        palette: str = "default",
        missing_color: Optional[str] = None,
        present_color: Optional[str] = None,
        show_legend: bool = True,
        max_label_length: int = 48,
    ) -> "_Density":
        """
        Overlapping KDE density curves split by missingness of ``missing_column``.

        Shows how the distribution of ``column`` differs between rows where
        ``missing_column`` is present (!NA) versus missing (NA).
        Diverging distributions suggest MAR or MNAR; overlapping suggests MCAR.

        Parameters
        ----------
        column : str
            Numeric column whose distribution is estimated.
        missing_column : str
            Column whose missingness splits the rows into the two curves.
        title : str, optional
            Title drawn above the figure. Also names the PNG that the toolbar's
            download button writes.
        width : int, default 900
            Figure width in pixels. Capped at 2000.
        height : int, default 520
            Figure height in pixels. Capped at 1000.
        background_color : str, optional
            Paper and plot background colour. ``None`` keeps the plotly default.
        text_color : str, optional
            Font colour for the figure. ``None`` keeps the plotly default.
        palette : {"default", "safe", "grayscale"}, default "default"
            Named colour pair for the present/missing distinction. ``"default"`` is red
            against green, which is the worst case for red-green colour vision
            deficiency and measures 1.48:1 in normal vision. ``"safe"`` and
            ``"grayscale"`` both clear the 3:1 WCAG floor for non-text content under
            simulated deuteranopia, protanopia and tritanopia; ``"grayscale"`` also
            survives monochrome printing.
        missing_color : str, optional
            Colour used to draw missing values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        present_color : str, optional
            Colour used to draw present values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        show_legend : bool, default True
            Whether to draw the legend.
        max_label_length : int, default 48
            Axis labels longer than this are truncated with an ellipsis, then
            de-duplicated if truncation made two labels identical. 0 falls back to a
            budget derived from ``width``.

        Returns
        -------
        _Density

        Notes
        -----
        A group with no spread (a constant column) gives a singular covariance matrix
        that a Gaussian KDE cannot invert, so that group falls back to a histogram.
        """
        from missingfcup.plots._density import _Density

        return _Density(
            data=_md(self),
            column=column,
            missing_column=missing_column,
            title=title,
            width=width,
            height=height,
            background_color=background_color,
            text_color=text_color,
            palette=palette,
            missing_color=missing_color,
            present_color=present_color,
            show_legend=show_legend,
            max_label_length=max_label_length,
        )

    def heatmap(
        self,
        *,
        kind: Literal["correlation", "direction", "dependence"] = "correlation",
        selected_columns: Optional[List[str]] = None,
        high_missingness_threshold: Optional[float] = None,
        show_values: bool = True,
        max_columns: int = 0,  # 0 = show all variables by default
        drop_constant_columns: bool = False,
        sort_by: Optional[Literal["missingness", "alphabetical"]] = "missingness",
        ascending: bool = False,
        show_upper_triangle: bool = False,
        selected_value_columns: Optional[List[str]] = None,
        selected_missing_columns: Optional[List[str]] = None,
        show_legend: bool = True,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        palette: str = "default",
        missing_color: Optional[str] = None,
        present_color: Optional[str] = None,
        max_label_length: int = 48,
    ) -> "_Plot":
        """
        Create a missingness association heatmap.

        ``kind="correlation"`` (default) reads the missingness mask alone and shows
        which columns tend to miss together. The other two read the observed
        *values* of one column against the missingness of another, which is the MAR
        question: ``kind="direction"`` gives the signed point-biserial association,
        so it says which way a relationship runs; ``kind="dependence"`` gives an
        unsigned 0-1 distance from independence, so it also sees relationships with
        no direction, such as gaps concentrated at both tails of a column at once.

        ``"correlation"`` cannot separate MAR from MCAR, because it never reads a
        value. Only the two value kinds can.

        ``selected_value_columns`` / ``selected_missing_columns`` apply to
        Value kinds only.

        Parameters
        ----------
        kind : {"correlation", "direction", "dependence"}, default "correlation"
            Which association to compute. Selects the plot class that is built.
            ``"correlation"`` is symmetric and signed, over the missingness mask.
            ``"direction"`` is asymmetric and signed, values against missingness.
            ``"dependence"`` is asymmetric and unsigned, on a 0-1 scale, with the
            statistic chosen by each value column's dtype: Kolmogorov-Smirnov for
            numeric columns and Cramer's V for categorical ones.
        selected_columns : list of str, optional
            Restrict the plot to these columns. Applied after the
            ``high_missingness_threshold`` filter, so a column dropped there stays
            dropped. Raises ValueError if nothing survives.
        high_missingness_threshold : float, optional
            Drop columns whose missing rate reaches this value, before anything
            else. ``None``, the default, keeps every column: the emptiest ones are
            usually what a missing-data plot is for.
        show_values : bool, default True
            Write each value into its cell. Suppressed automatically when the matrix is
            larger than 30 on either side, where the text would be unreadable.
        max_columns : int, default 0
            Cap on how many columns are drawn. 0 means show all.
        drop_constant_columns : bool, default False
            Drop columns whose missingness never varies. Their association with
            anything is undefined, so they draw a row and a column of blank cells.
            Off by default because that includes every column with no missing
            values, and hiding those hides how much of the data is intact.
        sort_by : {"missingness", "alphabetical"}, default "missingness"
            What to order the columns on. ``None`` keeps the DataFrame's own
            order. Pairs with ``ascending``, as in pandas.
        ascending : bool, default False
            Sort direction. The default puts the emptiest columns first.
        show_upper_triangle : bool, default False
            Mask the lower triangle and draw only the upper one, which drops the
            mirrored duplicates a symmetric matrix carries. ``kind="correlation"``
            only; raises for the value kinds, whose axes mean different things so
            nothing in them is a duplicate.
        selected_value_columns : list of str, optional
            Columns whose *values* form one axis. Value kinds only; falls back to
            ``selected_columns``. Passing it for ``"correlation"`` raises ValueError.
        selected_missing_columns : list of str, optional
            Columns whose *missingness* forms the other axis. Value kinds only; falls
            back to ``selected_columns``.
        show_legend : bool, default True
            Whether to draw the colour key beside the plot.
        title : str, optional
            Title drawn above the figure. Also names the PNG that the toolbar's
            download button writes.
        width : int, default 900
            Figure width in pixels. Capped at 2000.
        height : int, default 520
            Figure height in pixels. Capped at 1000.
        background_color : str, optional
            Paper and plot background colour. ``None`` keeps the plotly default.
        text_color : str, optional
            Font colour for the figure. ``None`` keeps the plotly default.
        palette : {"default", "safe", "grayscale"}, default "default"
            Named colour pair for the present/missing distinction. ``"default"`` is red
            against green, which is the worst case for red-green colour vision
            deficiency and measures 1.48:1 in normal vision. ``"safe"`` and
            ``"grayscale"`` both clear the 3:1 WCAG floor for non-text content under
            simulated deuteranopia, protanopia and tritanopia; ``"grayscale"`` also
            survives monochrome printing.
        missing_color : str, optional
            Colour used to draw missing values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        present_color : str, optional
            Colour used to draw present values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        max_label_length : int, default 48
            Axis labels longer than this are truncated with an ellipsis, then
            de-duplicated if truncation made two labels identical. 0 falls back to a
            budget derived from ``width``.

        Returns
        -------
        _Plot
            A ``_HeatmapCorrelation``, ``_HeatmapDirection`` or ``_HeatmapDependence``.

        Raises
        ------
        ValueError
            If ``kind`` is not one of the three, or if the value-side column options
            are given for ``kind="correlation"``.
        """
        if kind == "predictive":
            # Named explicitly because notebooks and scripts written against the
            # pre-release surface still pass it. It correlated "column i is present"
            # against "column j is missing", which is the correlation kind with its
            # sign flipped: the two masks are one fact, so corr(1 - miss_i, miss_j)
            # == -corr(miss_i, miss_j) for every pair. Same grid, opposite poles.
            raise ValueError(
                "kind='predictive' has been removed. It computed the 'correlation' "
                "kind negated -- presence and missingness are the same mask, so it "
                "carried no separate information. Use kind='correlation' for which "
                "columns share a gap pattern, or kind='direction' for whether "
                "observed values explain the gaps, which is the MAR question "
                "'predictive' was filed under but never answered."
            )
        if kind == "biserial":
            # Renamed rather than kept as an alias: "biserial" named the statistic,
            # and the same grid now offers two, so the name no longer identifies
            # which one you would get.
            raise ValueError(
                "kind='biserial' is now kind='direction'. The name described the "
                "statistic rather than the question, and this grid can now answer it "
                "two ways: kind='direction' keeps the signed point-biserial reading, "
                "and kind='dependence' measures any departure from independence, "
                "including gaps at both tails of a column, which the signed "
                "statistic reports as zero."
            )
        if kind not in _VALUE_KINDS + ("correlation",):
            raise ValueError(
                f"kind must be 'correlation', 'direction' or 'dependence', got {kind!r}"
            )
        if kind not in _VALUE_KINDS and (
            selected_value_columns is not None or selected_missing_columns is not None
        ):
            raise ValueError(
                "selected_value_columns / selected_missing_columns are only valid "
                "for kind='direction' and kind='dependence'"
            )
        if kind in _VALUE_KINDS and show_upper_triangle:
            # Mask exists because a symmetric matrix says everything twice. The value
            # kinds say nothing twice: axes are value columns against missing columns,
            # so value(a) vs missing(b) and value(b) vs missing(a) are different
            # questions. Masking by position deletes real associations.
            raise ValueError(
                f"show_upper_triangle applies to the symmetric heatmaps only, not "
                f"kind={kind!r}. Its axes carry different meanings (values on one, "
                f"missingness on the other), so no cell mirrors another and hiding "
                f"half of them would drop real associations. Use "
                f"selected_value_columns / selected_missing_columns to narrow it."
            )

        shared = dict(
            data=_md(self),
            selected_columns=selected_columns,
            high_missingness_threshold=high_missingness_threshold,
            show_values=show_values,
            max_columns=max_columns,
            drop_constant_columns=drop_constant_columns,
            sort_by=sort_by,
            ascending=ascending,
            show_upper_triangle=show_upper_triangle,
            show_legend=show_legend,
            title=title,
            width=width,
            height=height,
            background_color=background_color,
            text_color=text_color,
            palette=palette,
            missing_color=missing_color,
            present_color=present_color,
            max_label_length=max_label_length,
        )
        if kind == "correlation":
            from missingfcup.plots._heatmap_correlation import _HeatmapCorrelation

            return _HeatmapCorrelation(**shared)
        from missingfcup.plots._heatmap_value import _HeatmapDependence, _HeatmapDirection

        value_kind = _HeatmapDirection if kind == "direction" else _HeatmapDependence
        return value_kind(
            **shared,
            selected_value_columns=selected_value_columns,
            selected_missing_columns=selected_missing_columns,
        )

    def rate(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        high_missingness_threshold: Optional[float] = None,
        measure: Literal["fraction", "percentage"] = "fraction",
        show_values: bool = True,
        max_columns: int = 0,
        sort_by: Optional[Literal["missingness", "alphabetical"]] = "missingness",
        ascending: bool = False,
        show_legend: bool = True,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        palette: str = "default",
        missing_color: Optional[str] = None,
        present_color: Optional[str] = None,
        max_label_length: int = 48,
    ) -> "_Rate":
        """
        Create a single-row heatmap of missing rates per column.

        One row of cells, one per column, shaded by missing rate. This stays readable
        on wide datasets where a bar chart would get crowded.

        Parameters
        ----------
        selected_columns : list of str, optional
            Restrict the plot to these columns. Applied after the
            ``high_missingness_threshold`` filter, so a column dropped there stays
            dropped. Raises ValueError if nothing survives.
        high_missingness_threshold : float, optional
            Drop columns whose missing rate reaches this value, before anything
            else. ``None``, the default, keeps every column: the emptiest ones are
            usually what a missing-data plot is for.
        measure : {"fraction", "percentage"}, default "fraction"
            Whether the rate is drawn on a 0-1 or a 0-100 scale. There is no
            ``"count"`` here: a rate strip is the share by definition.
        show_values : bool, default True
            Write the rate into each cell. Automatically suppressed past 20 columns,
            where the numbers would overlap into an unreadable smear.
        max_columns : int, default 0
            Cap on how many columns are drawn. 0 means no cap.
        sort_by : {"missingness", "alphabetical"}, default "missingness"
            What to order the columns on. ``None`` keeps the DataFrame's own
            order. Pairs with ``ascending``, as in pandas.
        ascending : bool, default False
            Sort direction. The default puts the emptiest columns first.
        show_legend : bool, default True
            Whether to draw the colour key beside the plot.
        title : str, optional
            Title drawn above the figure. Also names the PNG that the toolbar's
            download button writes.
        width : int, default 900
            Figure width in pixels. Capped at 2000.
        height : int, default 520
            Figure height in pixels. Capped at 1000.
        background_color : str, optional
            Paper and plot background colour. ``None`` keeps the plotly default.
        text_color : str, optional
            Font colour for the figure. ``None`` keeps the plotly default.
        palette : {"default", "safe", "grayscale"}, default "default"
            Named colour pair for the present/missing distinction. ``"default"`` is red
            against green, which is the worst case for red-green colour vision
            deficiency and measures 1.48:1 in normal vision. ``"safe"`` and
            ``"grayscale"`` both clear the 3:1 WCAG floor for non-text content under
            simulated deuteranopia, protanopia and tritanopia; ``"grayscale"`` also
            survives monochrome printing.
        missing_color : str, optional
            Colour used to draw missing values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        present_color : str, optional
            Colour used to draw present values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        max_label_length : int, default 48
            Axis labels longer than this are truncated with an ellipsis, then
            de-duplicated if truncation made two labels identical. 0 falls back to a
            budget derived from ``width``.

        Returns
        -------
        _Rate
        """
        from missingfcup.plots._rate import _Rate

        return _Rate(
            data=_md(self),
            selected_columns=selected_columns,
            high_missingness_threshold=high_missingness_threshold,
            measure=measure,
            show_values=show_values,
            max_columns=max_columns,
            sort_by=sort_by,
            ascending=ascending,
            show_legend=show_legend,
            title=title,
            width=width,
            height=height,
            background_color=background_color,
            text_color=text_color,
            palette=palette,
            missing_color=missing_color,
            present_color=present_color,
            max_label_length=max_label_length,
        )

    def dendrogram(
        self,
        *,
        selected_columns: Optional[List[str]] = None,
        high_missingness_threshold: Optional[float] = None,
        max_columns: int = 0,
        drop_constant_columns: bool = False,
        linkage: Literal[
            "single", "complete", "average", "weighted", "centroid", "median", "ward"
        ] = "average",
        use_abs_correlation: bool = False,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        palette: str = "default",
        missing_color: Optional[str] = None,
        present_color: Optional[str] = None,
        max_label_length: int = 48,
    ) -> "_Dendrogram":
        """
        Create a dendrogram of missingness correlation between columns.

        Columns joined low in the tree tend to go missing together. Distance is
        ``1 - correlation`` between the columns' missingness masks.

        Parameters
        ----------
        selected_columns : list of str, optional
            Restrict the plot to these columns. Applied after the
            ``high_missingness_threshold`` filter, so a column dropped there stays
            dropped. Raises ValueError if nothing survives.
        high_missingness_threshold : float, optional
            Drop columns whose missing rate reaches this value, before anything
            else. ``None``, the default, keeps every column: the emptiest ones are
            usually what a missing-data plot is for.
        max_columns : int, default 0
            Cap on how many columns are clustered. 0 means no cap.
        drop_constant_columns : bool, default False
            Drop columns whose missingness never varies. Their correlation is
            undefined, so clustering them says nothing. Off by default because
            that includes every column with no missing values.
        linkage : {"single", "complete", "average", "weighted", "centroid", "median", "ward"}, default "average"
            Linkage criterion passed to ``scipy.cluster.hierarchy.linkage``, which
            decides how the distance between two clusters is measured.
        use_abs_correlation : bool, default False
            Cluster on the absolute correlation, so a strongly negative relationship
            counts as close rather than far apart.
        title : str, optional
            Title drawn above the figure. Also names the PNG that the toolbar's
            download button writes.
        width : int, default 900
            Figure width in pixels. Capped at 2000.
        height : int, default 520
            Figure height in pixels. Capped at 1000.
        background_color : str, optional
            Paper and plot background colour. ``None`` keeps the plotly default.
        text_color : str, optional
            Font colour for the figure. ``None`` keeps the plotly default.
        palette : {"default", "safe", "grayscale"}, default "default"
            Named colour pair for the present/missing distinction. ``"default"`` is red
            against green, which is the worst case for red-green colour vision
            deficiency and measures 1.48:1 in normal vision. ``"safe"`` and
            ``"grayscale"`` both clear the 3:1 WCAG floor for non-text content under
            simulated deuteranopia, protanopia and tritanopia; ``"grayscale"`` also
            survives monochrome printing.
        missing_color : str, optional
            Colour used to draw missing values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        present_color : str, optional
            Colour used to draw present values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        max_label_length : int, default 48
            Axis labels longer than this are truncated with an ellipsis, then
            de-duplicated if truncation made two labels identical. 0 falls back to a
            budget derived from ``width``.

        Returns
        -------
        _Dendrogram

        Raises
        ------
        ValueError
            If fewer than two columns with varying missingness survive the filters.
        """
        from missingfcup.plots._dendrogram import _Dendrogram

        return _Dendrogram(
            data=_md(self),
            selected_columns=selected_columns,
            high_missingness_threshold=high_missingness_threshold,
            max_columns=max_columns,
            drop_constant_columns=drop_constant_columns,
            linkage=linkage,
            use_abs_correlation=use_abs_correlation,
            title=title,
            width=width,
            height=height,
            background_color=background_color,
            text_color=text_color,
            palette=palette,
            missing_color=missing_color,
            present_color=present_color,
            max_label_length=max_label_length,
        )

    def boxplot(
        self,
        column: str,
        missing_column: str,
        *,
        shape: Literal["box", "violin"] = "box",
        title: Optional[str] = None,
        width: int = 900,
        height: int = 520,
        background_color: Optional[str] = None,
        text_color: Optional[str] = None,
        palette: str = "default",
        missing_color: Optional[str] = None,
        present_color: Optional[str] = None,
        show_legend: bool = True,
        max_label_length: int = 48,
        kind: None = None,
    ) -> "_Boxplot":
        """
        Create a box (or violin) plot comparing the distribution of ``column``
        between rows where ``missing_column`` is present vs. missing.

        Useful for diagnosing MAR and MNAR:
        * Different distributions: missingness of ``missing_column`` may relate to values of ``column``
        * Similar distributions: consistent with MCAR

        Parameters
        ----------
        column : str
            Column whose value distribution to plot on the y-axis.
        missing_column : str
            Column whose missingness splits the two groups (present vs. missing).
        shape : {"box", "violin"}, default "box"
            ``"box"`` shows medians and quartiles; ``"violin"`` also shows the shape of
            each distribution. Both draw the same numbers, so this selects how the
            distribution is rendered rather than what is computed. It is named
            ``shape`` and not ``kind`` for that reason: every ``kind`` in this package
            changes the result, and this one cannot.
        kind : None
            The former name of ``shape``. It is accepted only so that passing it
            raises a message naming the replacement, rather than the bare TypeError
            an unknown keyword would give. Passing anything but ``None`` is an
            error; it takes no other value and will be removed.
        title : str, optional
            Title drawn above the figure. Also names the PNG that the toolbar's
            download button writes.
        width : int, default 900
            Figure width in pixels. Capped at 2000.
        height : int, default 520
            Figure height in pixels. Capped at 1000.
        background_color : str, optional
            Paper and plot background colour. ``None`` keeps the plotly default.
        text_color : str, optional
            Font colour for the figure. ``None`` keeps the plotly default.
        palette : {"default", "safe", "grayscale"}, default "default"
            Named colour pair for the present/missing distinction. ``"default"`` is red
            against green, which is the worst case for red-green colour vision
            deficiency and measures 1.48:1 in normal vision. ``"safe"`` and
            ``"grayscale"`` both clear the 3:1 WCAG floor for non-text content under
            simulated deuteranopia, protanopia and tritanopia; ``"grayscale"`` also
            survives monochrome printing.
        missing_color : str, optional
            Colour used to draw missing values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        present_color : str, optional
            Colour used to draw present values. Overrides the palette for this colour
            only; ``None``, the default, takes it from ``palette``.
        show_legend : bool, default True
            Whether to draw the legend.
        max_label_length : int, default 48
            Axis labels longer than this are truncated with an ellipsis, then
            de-duplicated if truncation made two labels identical. 0 falls back to a
            budget derived from ``width``.

        Returns
        -------
        _Boxplot

        Example
        -------
        md.boxplot(column="fare", missing_column="age")
        answers "Do passengers with missing age tend to pay different fares?"
        """
        if kind is not None:
            # Accepted purely to say this. Every other `kind` in the package selects
            # what gets computed, and box against violin selects neither the data nor
            # the statistic, so it was the one that taught the wrong expectation.
            raise ValueError(
                f"boxplot(kind={kind!r}) is now boxplot(shape={kind!r}). The option "
                f"chooses how one result is drawn, not which result is computed, and "
                f"`kind` means the latter everywhere else in this package: "
                f"heatmap(kind=) picks the statistic, parallel_coordinates(kind=) "
                f"picks between values and the missingness mask."
            )

        from missingfcup.plots._boxplot import _Boxplot

        return _Boxplot(
            data=_md(self),
            column=column,
            missing_column=missing_column,
            shape=shape,
            title=title,
            width=width,
            height=height,
            background_color=background_color,
            text_color=text_color,
            palette=palette,
            missing_color=missing_color,
            present_color=present_color,
            show_legend=show_legend,
            max_label_length=max_label_length,
        )
