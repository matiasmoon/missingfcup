"""Plot tests.

These assert what ends up *inside* the figure: the values encoded in the traces, the
number of series, the axis configuration. A test that only checks ``.fig is not None``
passes even when the figure is wrong, so it is not worth writing.

The fixture is the packaged sample dataset, so these tests and the examples exercise
the same data.
"""

import numpy as np
import pandas as pd
import pytest

import missingfcup as mf
from missingfcup import MissingData, Panel


@pytest.fixture
def df():
    return mf.sample_data()


@pytest.fixture
def md(df):
    return MissingData(df)


# Every plot builds and draws something

# (name, builder) for each public plot, including the dispatched variants of
# bar() and heatmap(). Anything added to the plot surface belongs here.
PLOTS = [
    ("matrix", lambda md: md.matrix()),
    ("bar_count", lambda md: md.bar()),
    ("bar_rate", lambda md: md.bar(measure="fraction")),
    ("rate", lambda md: md.rate()),
    ("totals", lambda md: md.totals()),
    ("heatmap_correlation", lambda md: md.heatmap()),
    ("heatmap_predictive", lambda md: md.heatmap(kind="predictive")),
    ("heatmap_biserial", lambda md: md.heatmap(kind="biserial")),
    ("dendrogram", lambda md: md.dendrogram()),
    ("venn", lambda md: md.venn(selected_columns=["age", "income", "score"])),
    ("upset", lambda md: md.upset(selected_columns=["age", "income", "score", "rating"])),
    ("scatterplot", lambda md: md.scatterplot(x="age", y="income")),
    ("density", lambda md: md.density(column="income", missing_column="age")),
    ("boxplot", lambda md: md.boxplot(column="income", missing_column="age")),
    ("parallel_coordinates", lambda md: md.parallel_coordinates(missing_column="age")),
]


@pytest.mark.parametrize("name,build", PLOTS, ids=[n for n, _ in PLOTS])
def test_plot_draws_at_least_one_non_empty_trace(md, name, build):
    """Every plot must produce traces, and those traces must carry data."""
    fig = build(md).fig
    assert len(fig.data) > 0, f"{name} produced no traces"
    assert any(
        getattr(trace, attr, None) is not None and len(getattr(trace, attr)) > 0
        for trace in fig.data
        for attr in ("x", "y", "z")
    ), f"{name} produced only empty traces"


# What the plots actually encode


def test_matrix_encodes_the_missingness_mask(md, df):
    """Binary mode draws 1 where a cell is present and 0 where it is missing."""
    z = np.asarray(md.matrix().fig.data[0].z)
    expected = (~df.isna()).astype(int).to_numpy()
    assert z.shape == expected.shape
    assert np.array_equal(z, expected)


def test_bar_count_encodes_missing_counts_per_column(md, df):
    bar = md.bar().fig.data[0]
    drawn = dict(zip(bar.x, bar.y))
    for col, expected in df.isna().sum().items():
        assert drawn[col] == expected


def test_bar_show_both_stacks_present_and_missing(md, df):
    fig = md.bar(show_both=True).fig
    assert len(fig.data) == 2
    assert fig.layout.barmode == "stack"
    missing, present = fig.data[0], fig.data[1]
    # Each column's two bars must add up to the row count.
    for m, p in zip(missing.y, present.y):
        assert m + p == len(df)


@pytest.mark.parametrize("measure", ["fraction", "percentage"])
def test_bar_show_both_rejects_the_rate_measures(md, measure):
    """It used to be dropped quietly, drawing a plain rate bar that looked like the
    argument had been honoured."""
    with pytest.raises(ValueError, match="show_both applies to measure='count' only"):
        md.bar(measure=measure, show_both=True)


def test_bar_selects_columns_the_way_every_other_column_plot_does(md):
    """bar used to carry its own completeness filter alongside the shared options.
    The shared ones cover it, so the same call shape has to work here too."""
    drawn = list(md.bar(high_missingness_threshold=0.1).fig.data[0].x)
    assert drawn == ["visits"], "only the fully complete column is under 0.1 missing"

    capped = list(md.bar(sort_by="missingness", ascending=False, max_columns=2).fig.data[0].x)
    assert capped == ["income", "age"], "the two emptiest columns, emptiest first"


@pytest.mark.parametrize(
    "name,build",
    [
        ("bar", lambda md: md.bar()),
        ("bar_rate", lambda md: md.bar(measure="fraction")),
        ("matrix", lambda md: md.matrix()),
        ("rate", lambda md: md.rate()),
        ("heatmap", lambda md: md.heatmap()),
        ("dendrogram", lambda md: md.dendrogram()),
    ],
)
def test_a_near_empty_column_is_shown_by_default(name, build):
    """The emptiest column is the one a missing-data plot most needs to show, so
    high_missingness_threshold is opt-in rather than on by default."""
    frame = pd.DataFrame(
        {
            "a": [1.0, None, 3.0, 4.0, 5.0],
            "b": [1.0, 2.0, None, 4.0, 5.0],
            "empty": [None] * 5,
        }
    )
    md = MissingData(frame)
    plot = build(md)

    drawn = set()
    ticks = getattr(plot.fig.layout.xaxis, "ticktext", None)
    if ticks:
        drawn |= {t for t in ticks if isinstance(t, str)}
    for trace in plot.fig.data:
        values = getattr(trace, "x", None)
        if values is not None and len(values) and isinstance(values[0], str):
            drawn |= {v for v in values if isinstance(v, str)}

    assert "empty" in drawn, f"{name} hid the fully-missing column by default"


def test_high_missingness_threshold_is_opt_in(md):
    """One parameter now: a threshold filters, None (the default) does not."""
    kept = set(md.bar().fig.data[0].x)
    filtered = set(md.bar(high_missingness_threshold=0.1).fig.data[0].x)
    assert filtered < kept, "a low threshold should drop columns the default keeps"


@pytest.mark.parametrize(
    "measure",
    ["count", "fraction", "percentage"],
)
def test_venn_measure_matches_bar(md, measure):
    """venn uses the same measure vocabulary as bar, so the same call shape gives
    the same kind of number."""
    kwargs, expected = {"measure": measure}, measure
    cols = ["age", "income", "score"]
    venn = np.asarray(md.venn(selected_columns=cols, **kwargs).fig.data[0].y, dtype=float)
    counts = np.asarray(md.venn(selected_columns=cols).fig.data[0].y, dtype=float)
    rows = len(md.data)

    if expected == "count":
        assert np.allclose(venn, counts)
    elif expected == "fraction":
        assert np.allclose(venn, counts / rows)
    else:
        assert np.allclose(venn, counts / rows * 100)


@pytest.mark.parametrize(
    "cols,message",
    [
        (None, "selected_columns"),
        (["age", "income"], "exactly 3"),
        (["age", "income", "score", "rating"], "exactly 3"),
    ],
)
def test_venn_requires_exactly_three_named_columns(md, cols, message):
    """A 3-set Venn has 7 exclusive regions, so no other number defines the plot.
    Choosing the columns for the caller would answer a different question."""
    plot = md.venn(selected_columns=cols)
    with pytest.raises(ValueError, match=message):
        _ = plot.fig


@pytest.mark.parametrize("measure", ["count", "fraction", "percentage"])
def test_upset_measure_matches_venn(md, measure):
    """They draw the same quantity for the same columns, so the same option has to
    give the same numbers. upset used to be count-only, losing a reading the moment
    a user outgrew venn's three columns."""
    cols = ["age", "income", "score"]
    upset = md.upset(selected_columns=cols, measure=measure).fig.data[0]
    venn = md.venn(selected_columns=cols, measure=measure).fig.data[0]
    # The largest region is the same rows counted the same way in both plots.
    assert max(float(v) for v in upset.y) == pytest.approx(max(float(v) for v in venn.y))


def test_upset_measure_scales_both_panels(md):
    """Intersection bars and set-size bars share one scale, so scaling one and not
    the other would make them silently incomparable."""
    cols = ["age", "income", "score"]
    counts = md.upset(selected_columns=cols).fig
    fractions = md.upset(selected_columns=cols, measure="fraction").fig
    rows = len(md.data)

    assert float(fractions.data[0].y[0]) == pytest.approx(float(counts.data[0].y[0]) / rows)
    assert float(fractions.data[1].x[0]) == pytest.approx(float(counts.data[1].x[0]) / rows)


def axes_of(md, **kwargs):
    """The axis labels parallel_coordinates drew, left to right."""
    return list(md.parallel_coordinates(**kwargs).fig.layout.xaxis.ticktext)


def test_parallel_coordinates_orders_its_axes(md):
    """A relationship is visible only between adjacent axes, so axis order is what
    the plot can show, not decoration."""
    assert axes_of(md) == list(md.columns), "the default keeps the frame's order"
    assert axes_of(md, sort_by="missingness") == list(
        md.col_missing_rate.sort_values(ascending=False).index
    )
    assert axes_of(md, sort_by="alphabetical", ascending=True) == sorted(md.columns)


def test_parallel_coordinates_shares_the_column_filters(md):
    """It had selected_columns and max_columns but not the threshold, which is the
    one that decides whether the near-empty columns are drawn at all."""
    assert axes_of(md, high_missingness_threshold=0.9) == list(md.columns)
    assert axes_of(md, sort_by="missingness", max_columns=3) == list(
        md.col_missing_rate.sort_values(ascending=False).index[:3]
    )


def test_totals_can_turn_its_values_off(md):
    """Every other value-drawing plot allows it; totals was the one that did not."""
    assert md.totals(show_values=True).fig.data[0].text is not None
    assert md.totals(show_values=False).fig.data[0].text is None


def test_bar_raises_when_no_selected_column_exists(md):
    """bar used to draw an empty chart instead, unlike every other column plot."""
    plot = md.bar(selected_columns=["nope"])
    with pytest.raises(ValueError, match="selected_columns"):
        _ = plot.fig


def test_bar_max_columns_zero_means_no_cap(md):
    """0 means 'no cap' on the other column plots; bar drew nothing at all."""
    assert len(md.bar(max_columns=0).fig.data[0].x) == len(md.columns)


@pytest.mark.parametrize(
    "build",
    [
        lambda md: md.bar(measure="fraction"),
        lambda md: md.bar(measure="percentage"),
        lambda md: md.rate(),
        lambda md: md.venn(selected_columns=["age", "income", "score"], measure="fraction"),
    ],
)
def test_rates_are_drawn_with_two_decimals(md, build):
    """One format for every rate the package prints, so the same number reads the
    same way whichever plot drew it."""
    text = build(md).fig.data[0].text
    # rate() is a one-row heatmap, so its text is nested one level deeper.
    flat = [t for row in text for t in row] if isinstance(text[0], list) else list(text)
    drawn = [t for t in flat if t]
    assert drawn
    for label in drawn:
        assert len(label.rstrip("%").split(".")[1]) == 2, label


def test_bar_rate_percentage_is_fraction_times_100(md):
    fraction = md.bar(measure="fraction").fig.data[0].y
    percentage = md.bar(measure="percentage").fig.data[0].y
    assert np.allclose(np.asarray(percentage), np.asarray(fraction) * 100)


def test_rate_strip_is_one_row_of_missing_rates(md):
    z = np.asarray(md.rate().fig.data[0].z)
    assert z.shape[0] == 1, "rate() draws a single row"
    assert ((z >= 0) & (z <= 1)).all()


def test_rate_drops_in_cell_values_once_they_would_not_fit():
    """The strip is one row, so past ~20 columns the numbers collide into a smear.
    The cap is fixed rather than an option: it is a legibility limit, not a taste."""
    wide = pd.DataFrame({f"c{i}": [1.0, None, 3.0] for i in range(40)})
    narrow = pd.DataFrame({f"c{i}": [1.0, None, 3.0] for i in range(5)})

    assert MissingData(narrow).rate().fig.data[0].texttemplate == "%{text}"
    assert MissingData(wide).rate().fig.data[0].texttemplate is None
    # show_values=False still suppresses them at any width.
    assert MissingData(narrow).rate(show_values=False).fig.data[0].texttemplate is None


def test_heatmap_correlation_diagonal_is_one(md):
    """Every column correlates perfectly with itself.

    Constant columns are excluded: a column that is never missing has no variance in
    its mask, so its self-correlation is NaN rather than 1.
    """
    fig = md.heatmap(drop_constant_columns=True).fig
    heat = fig.data[-1]  # last trace holds the values; an earlier one masks NaNs
    z = np.asarray(heat.z, dtype=float)
    assert len(heat.x) > 1, "need at least two varying columns to be meaningful"
    for i, label in enumerate(heat.x):
        assert np.isclose(z[i][i], 1.0), f"self-correlation of {label} is not 1"


def test_heatmap_keeps_constant_columns_as_nan_by_default(md):
    """A never-missing column has no missingness variance, so it reads as NaN. It is
    still drawn: hiding it would hide how much of the data is intact."""
    heat = md.heatmap().fig.data[-1]
    z = np.asarray(heat.z, dtype=float)
    visits = list(heat.x).index("visits")  # the complete column in sample_data
    assert np.isnan(z[visits][visits])


def test_heatmap_drops_constant_columns_when_asked(md):
    """Opting in has to actually remove them, not just blank their cells."""
    assert "visits" not in list(md.heatmap(drop_constant_columns=True).fig.data[-1].x)


def test_heatmap_biserial_rejects_the_triangle_mask(md):
    """The mask drops the mirrored half of a symmetric matrix. Biserial is not
    symmetric, so masking by position deleted real associations instead."""
    with pytest.raises(ValueError, match="show_upper_triangle applies to the symmetric"):
        md.heatmap(kind="biserial", show_upper_triangle=True)


@pytest.mark.parametrize("kind", ["correlation", "predictive"])
def test_heatmap_triangle_mask_drops_only_duplicates(md, kind):
    """On a symmetric matrix every masked cell has a surviving mirror image."""
    full = np.asarray(md.heatmap(kind=kind).fig.data[-1].z, dtype=float)
    masked = np.asarray(md.heatmap(kind=kind, show_upper_triangle=True).fig.data[-1].z, dtype=float)
    dropped = np.isfinite(full) & ~np.isfinite(masked)
    assert dropped.any(), "the mask did nothing"
    assert np.isfinite(masked.T[dropped]).all(), "a cell was dropped with no mirror kept"


def test_heatmap_biserial_axes_carry_different_meanings(md):
    """Biserial is asymmetric: rows are value columns, columns are missingness."""
    fig = md.heatmap(kind="biserial").fig
    assert fig.layout.xaxis.title.text == "Missing column"
    assert fig.layout.yaxis.title.text == "Value column"


def test_venn_draws_seven_exclusive_subsets(md, df):
    cols = ["age", "income", "score"]
    bar = md.venn(selected_columns=cols).fig.data[0]
    assert len(bar.y) == 7, "a 3-set Venn has 7 exclusive regions"
    # Exclusive regions cannot overlap, so they sum to the rows missing at least
    # one of the three columns.
    assert sum(bar.y) == int(df[cols].isna().any(axis=1).sum())


def test_upset_intersection_sizes_match_the_data(md, df):
    cols = ["age", "income", "score"]
    fig = md.upset(selected_columns=cols).fig
    intersections = fig.data[0]
    patterns = df[cols].isna().apply(lambda row: tuple(row.index[row]), axis=1)
    expected = patterns[patterns.astype(bool)].value_counts()
    assert sorted(intersections.y, reverse=True) == sorted(expected.tolist(), reverse=True)


def test_scatterplot_plots_every_row_including_missing_ones(md, df):
    """A plain scatter would drop rows with a missing axis; this one offsets them."""
    fig = md.scatterplot(x="age", y="income").fig
    plotted = sum(len(trace.x) for trace in fig.data)
    assert plotted == len(df)


def test_density_draws_one_curve_per_missingness_group(md):
    fig = md.density(column="income", missing_column="age").fig
    assert len(fig.data) == 2
    assert {trace.name for trace in fig.data} == {"NA-age", "!NA-age"}


def test_boxplot_splits_rows_by_missingness_of_the_other_column(md, df):
    fig = md.boxplot(column="income", missing_column="age").fig
    assert len(fig.data) == 2
    present_expected = int(df.loc[~df["age"].isna(), "income"].notna().sum())
    missing_expected = int(df.loc[df["age"].isna(), "income"].notna().sum())
    assert len(fig.data[0].y) == present_expected
    assert len(fig.data[1].y) == missing_expected


def test_parallel_coordinates_draws_one_axis_per_column(md, df):
    fig = md.parallel_coordinates().fig
    assert list(fig.layout.xaxis.ticktext) == list(df.columns)


def test_parallel_coordinates_max_columns_caps_the_axes(md):
    capped = md.parallel_coordinates(max_columns=2).fig
    assert len(capped.layout.xaxis.ticktext) == 2


def test_dendrogram_draws_one_leaf_per_column(md, df):
    fig = md.dendrogram().fig
    assert len(fig.layout.xaxis.ticktext) == len(df.columns)
    # n leaves are joined by n-1 merges, one line trace each.
    assert len(fig.data) == len(df.columns) - 1


@pytest.mark.parametrize(
    "column,expected",
    [
        # Numbers sort numerically, strings lexicographically, and a categorical by
        # its declared categories -- which is how a caller states a custom order
        # without the package needing an option for it.
        (pd.Series([30.0, 10.0, None, 20.0]), ["10", "20", "30", "NaN"]),
        (pd.Series(["B", "C", None, "A"]), ["A", "B", "C", "NaN"]),
        (pd.Categorical(["M", "S", None, "L"], categories=["S", "M", "L"]), ["S", "M", "L", "NaN"]),
    ],
    ids=["numeric", "string", "categorical"],
)
def test_matrix_row_order_follows_the_column_dtype(column, expected):
    frame = pd.DataFrame({"key": column, "score": [1.0, None, 3.0, None]})
    drawn = MissingData(frame).matrix(sort_by="key", ascending=True).fig.data[0].y
    assert list(drawn) == expected


def test_matrix_sorts_missing_rows_last_in_both_directions():
    """Reversing the sort must not push the rows under study off the far end."""
    frame = pd.DataFrame({"key": [2.0, None, 1.0], "score": [1.0, None, 3.0]})
    md = MissingData(frame)
    assert list(md.matrix(sort_by="key", ascending=True).fig.data[0].y)[-1] == "NaN"
    assert list(md.matrix(sort_by="key", ascending=False).fig.data[0].y)[-1] == "NaN"


@pytest.fixture
def nominal():
    """A nominal categorical: no value is inherently first, so alphabetical is an
    arbitrary choice and only the caller can say what the order should be."""
    return MissingData(
        pd.DataFrame(
            {
                "country": ["Spain", "France", "Portugal", None, "France", "Spain", "Italy"],
                "score": [1.0, None, 3.0, 4.0, None, 6.0, 7.0],
            }
        )
    )


def test_sort_categories_draws_the_sequence_as_written(nominal):
    drawn = (
        nominal.matrix(sort_by="country", sort_categories=["Portugal", "France", "Spain"])
        .fig.data[0]
        .y
    )
    # Italy was not named, so it follows the named values; NaN stays last.
    assert list(drawn) == ["Portugal", "France", "France", "Spain", "Spain", "Italy", "NaN"]


def test_sort_categories_ignores_ascending(nominal):
    """The sequence already says which value is first, so honouring a direction on
    top of it would draw the reverse of what the caller wrote."""
    order = ["Portugal", "France", "Spain"]
    both = [
        list(nominal.matrix(sort_by="country", sort_categories=order, ascending=asc).fig.data[0].y)
        for asc in (True, False)
    ]
    assert both[0] == both[1]
    assert both[0][0] == "Portugal"


def test_sort_categories_reverses_by_reversing_the_sequence(nominal):
    drawn = (
        nominal.matrix(sort_by="country", sort_categories=["Spain", "France", "Portugal"])
        .fig.data[0]
        .y
    )
    assert list(drawn)[:2] == ["Spain", "Spain"]


def test_sort_categories_needs_sort_by_to_name_a_column(nominal):
    """Otherwise the order is accepted and quietly ignored: nothing is sorting the
    rows for it to apply to."""
    for sort_by in (None, "missingness", "alphabetical"):
        plot = nominal.matrix(sort_by=sort_by, sort_categories=["Portugal"])
        with pytest.raises(ValueError, match="sort_categories needs sort_by"):
            _ = plot.fig


def test_sort_categories_rejects_values_that_are_not_there(nominal):
    """A typo would otherwise reorder nothing and say nothing."""
    plot = nominal.matrix(sort_by="country", sort_categories=["Narnia"])
    with pytest.raises(ValueError, match="None of sort_categories"):
        _ = plot.fig


def test_sort_categories_warns_about_values_that_are_not_there(nominal):
    """A partly-wrong list still draws, so it cannot raise -- the same order is worth
    reusing across datasets. But a typo looks identical from here, so it has to say
    something rather than silently ordering nothing."""
    plot = nominal.matrix(sort_by="country", sort_categories=["Portugal", "Narnia"])
    with pytest.warns(UserWarning, match=r"sort_categories named \['Narnia'\]"):
        drawn = plot.fig.data[0].y

    # The valid part of the order still applies.
    assert list(drawn)[0] == "Portugal"


def test_sort_categories_is_quiet_when_every_value_matches(nominal):
    """The warning must not fire on a correct call, including a deliberately
    partial order that names only some of the column's values."""
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        drawn = (
            nominal.matrix(sort_by="country", sort_categories=["Portugal", "France"]).fig.data[0].y
        )
    # Spain and Italy were not named, so they follow; nothing warned about them.
    assert list(drawn)[:3] == ["Portugal", "France", "France"]


def test_matrix_labels_rows_by_the_column_it_sorted_on(md):
    """Sorting scrambles the index, so labelling rows with it says nothing about
    where a row sits. The value that decided the position is what does."""
    ages = [v for v in md.data["age"] if pd.notna(v)]
    drawn = list(md.matrix(sort_by="age", ascending=False).fig.data[0].y)

    assert drawn[: len(ages)] == [str(int(v)) for v in sorted(ages, reverse=True)]
    assert drawn[-1] == "NaN", "rows missing that value still sort last"


def test_matrix_falls_back_to_the_index_with_no_sort_column(md):
    """The keyword sorts order columns, not rows, so there is no per-row value to
    label with and the index is all that is left."""
    for sort_by in (None, "missingness", "alphabetical"):
        drawn = list(md.matrix(sort_by=sort_by).fig.data[0].y)
        assert drawn == [str(i) for i in md.data.index]


def test_matrix_hover_labels_come_from_the_mask(md):
    """The label has to track the mask, not z, so it cannot drift out of step with
    however z happens to be encoded."""
    plot = md.matrix()
    text = np.asarray(plot.fig.data[0].text)
    mask = md.mask_missing.loc[:, plot._prepare_df().columns].to_numpy()

    assert set(text[mask]) == {"NA"}, "labelled a missing cell wrong"
    assert set(text[~mask]) == {"!NA"}, "labelled a present cell wrong"


def test_dendrogram_drops_constant_columns_when_asked(md):
    """Its distance to anything is undefined, so opting out of drawing it has to work."""
    assert "visits" not in list(md.dendrogram(drop_constant_columns=True).fig.layout.xaxis.ticktext)


# Options that change the output


def test_selected_columns_restricts_what_is_drawn(md):
    bar = md.bar(selected_columns=["age", "income"]).fig.data[0]
    assert list(bar.x) == ["age", "income"]


def test_sorting_the_rate_strip_acts_on_the_values(md):
    """Ordering acts on the values, not the labels.

    Tied columns keep their original relative order in both directions (the sort is
    stable), so asc is not simply desc reversed. Monotonicity is the real contract.
    """
    desc = np.asarray(md.rate(sort_by="missingness", ascending=False).fig.data[0].z)[0]
    asc = np.asarray(md.rate(sort_by="missingness", ascending=True).fig.data[0].z)[0]
    assert (np.diff(desc) <= 0).all(), "desc must not increase"
    assert (np.diff(asc) >= 0).all(), "asc must not decrease"
    assert sorted(desc) == sorted(asc), "same columns, different order"


def test_no_plot_sets_its_own_text_size(md):
    """Three plots used to hardcode three different sizes and only one of them was
    tunable. They all defer to the plotly default now, so the package is consistent
    with itself and with whatever theme the caller has set."""
    builds = [
        md.heatmap(),
        md.venn(selected_columns=["age", "income", "score"]),
        md.upset(selected_columns=["age", "income", "score"]),
        md.bar(),
        md.rate(),
        md.totals(),
    ]
    for plot in builds:
        for trace in plot.fig.data:
            font = getattr(trace, "textfont", None)
            assert font is None or font.size is None, f"{type(plot).__name__} sized its text"


def test_semantic_colors_are_configurable(md):
    fig = md.bar(missing_color="#123456").fig
    assert fig.data[0].marker.color == "#123456"


COLUMN_AXIS_PLOTS = [
    ("bar", lambda md: md.bar(), "Column"),
    ("bar_rate", lambda md: md.bar(measure="fraction"), "Column"),
    ("rate", lambda md: md.rate(), "Column"),
    ("venn", lambda md: md.venn(selected_columns=["age", "income", "score"]), "Missing columns"),
    ("dendrogram", lambda md: md.dendrogram(), "Column"),
    ("parallel_coordinates", lambda md: md.parallel_coordinates(), "Column"),
]


@pytest.mark.parametrize(
    "name,build,expected", COLUMN_AXIS_PLOTS, ids=[n for n, _, _ in COLUMN_AXIS_PLOTS]
)
def test_column_axis_has_a_generic_label(md, df, name, build, expected):
    """The x axis enumerates every column, so it must not be named after one of them.

    These six once labelled it with whichever column happened to sort first, which read
    as though the axis *was* that column.
    """
    title = build(md).fig.layout.xaxis.title.text
    assert title not in df.columns, f"{name}: axis titled with the column name {title!r}"
    assert title == expected


def test_bar_horizontal_swaps_the_axis_labels(md):
    """Rotating the bars must move the column label with them."""
    fig = md.bar(orientation="horizontal").fig
    assert fig.layout.yaxis.title.text == "Column"
    assert fig.layout.xaxis.title.text == "Missing rows"


def test_bar_value_axis_says_what_it_counts(md):
    """A bar alone is the missing count; stacked, its height is every row. "Count"
    named neither, so the axis did not say which of the two was drawn."""
    assert md.bar().fig.layout.yaxis.title.text == "Missing rows"
    assert md.bar(show_both=True).fig.layout.yaxis.title.text == "Rows"


# Errors


@pytest.mark.parametrize(
    "call,expected",
    [
        (lambda md: md.bar(measure="nope"), ValueError),
        (lambda md: md.heatmap(kind="nope"), ValueError),
        (lambda md: md.heatmap(kind="correlation", selected_value_columns=["age"]), ValueError),
        (lambda md: md.scatterplot(x="age", y="absent").fig, ValueError),
        (lambda md: md.boxplot(column="absent", missing_column="age").fig, ValueError),
        (lambda md: md.density(column="absent", missing_column="age").fig, ValueError),
        (lambda md: md.parallel_coordinates(selected_columns=["absent"]).fig, ValueError),
    ],
)
def test_invalid_arguments_raise(md, call, expected):
    with pytest.raises(expected):
        call(md)


def test_non_numeric_column_raises_with_guidance(md):
    """Value-reading plots must say how to fix a categorical column, not just fail."""
    frame = mf.sample_data()
    frame["label"] = ["x"] * len(frame)
    plot = MissingData(frame).boxplot(column="label", missing_column="age")
    with pytest.raises(TypeError, match="factorize"):
        plot._build_figure()


# Output: saving


def test_save_png_writes_a_real_image(md, tmp_path):
    """Guards the kaleido dependency, which is the part most likely to break."""
    out = tmp_path / "figure.png"
    md.matrix().save(str(out))
    assert out.exists()
    assert out.stat().st_size > 1000, "PNG is suspiciously small"
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"


def test_save_html_creates_missing_directories(md, tmp_path):
    out = tmp_path / "nested" / "dir" / "figure.html"
    md.matrix().save(str(out))
    assert out.exists() and "plotly" in out.read_text()


# Panel


def test_panel_carries_every_subplot_trace(md):
    matrix, bar = md.matrix(), md.bar()
    combined = Panel([matrix, bar])._create_combined_figure()
    assert len(combined.data) == len(matrix.fig.data) + len(bar.fig.data)


def test_panel_does_not_mutate_the_plots_it_holds(md):
    """Panel deep-copies traces; showing a panel must not alter the original figure."""
    plot = md.bar()
    before = plot.fig.data[0].showlegend
    Panel([plot])._create_combined_figure()
    assert plot.fig.data[0].showlegend == before


def test_panel_add_chains_and_enforces_its_limit(md):
    panel = Panel(max_plots=2).add(md.matrix()).add(md.bar())
    assert len(panel.plots) == 2
    with pytest.raises(ValueError, match="max_plots"):
        panel.add(md.rate())


def test_panel_with_no_plots_raises(md):
    with pytest.raises(ValueError):
        Panel()._create_combined_figure()


# Package surface


def test_flat_facade_matches_the_plot_methods():
    """The flat functions and the MissingData plot methods must not drift apart."""
    from missingfcup.core.mixins._plots import _MissingDataPlotMixin

    methods = {n for n in vars(_MissingDataPlotMixin) if not n.startswith("_")}
    assert set(mf._functional.__all__) == methods
    for name in mf._functional.__all__:
        assert hasattr(mf, name), f"{name} not exported from missingfcup"


def test_flat_function_matches_the_method_it_wraps(df):
    """mf.bar(df) and MissingData(df).bar() must encode the same thing."""
    flat = mf.bar(df, measure="fraction").fig.data[0]
    method = MissingData(df).bar(measure="fraction").fig.data[0]
    assert list(flat.x) == list(method.x)
    assert list(flat.y) == list(method.y)


def test_factory_and_class_defaults_agree():
    """A public factory method must not declare a default that disagrees with the
    plot class it builds. Guards the two surfaces against silently drifting apart."""
    import inspect

    from missingfcup.core.mixins._plots import _MissingDataPlotMixin
    from missingfcup.plots._boxplot import _Boxplot
    from missingfcup.plots._dendrogram import _Dendrogram
    from missingfcup.plots._density import _Density
    from missingfcup.plots._matrix import _Matrix
    from missingfcup.plots._parallel_coordinates import _ParallelCoordinates
    from missingfcup.plots._rate import _Rate
    from missingfcup.plots._scatterplot import _Scatterplot
    from missingfcup.plots._upset import _Upset
    from missingfcup.plots._venn import _Venn

    # Only the single-class factories; bar() and heatmap() dispatch across classes.
    factory_to_class = {
        "matrix": _Matrix,
        "rate": _Rate,
        "dendrogram": _Dendrogram,
        "scatterplot": _Scatterplot,
        "upset": _Upset,
        "venn": _Venn,
        "parallel_coordinates": _ParallelCoordinates,
        "boxplot": _Boxplot,
        "density": _Density,
    }
    empty = inspect.Parameter.empty
    for name, cls in factory_to_class.items():
        fparams = inspect.signature(getattr(_MissingDataPlotMixin, name)).parameters
        cparams = inspect.signature(cls.__init__).parameters
        for pname, fp in fparams.items():
            if pname in ("self", "kwargs") or fp.default is empty:
                continue
            if pname in cparams and cparams[pname].default is not empty:
                assert fp.default == cparams[pname].default, (
                    f"{name}(): default for '{pname}' ({fp.default!r}) disagrees "
                    f"with {cls.__name__} ({cparams[pname].default!r})"
                )


def test_ipython_display_renders_inline(md):
    """Lets `mf.matrix(df)` render in a notebook without an explicit .show()."""
    plot = md.matrix()
    called = {}
    plot.show = lambda: called.setdefault("shown", True)
    plot._ipython_display_()
    assert called.get("shown") is True


def test_version_is_a_string():
    assert isinstance(mf.__version__, str)
    assert mf.__version__.count(".") >= 1


# sample_data


def test_sample_data_shape_and_gaps():
    frame = mf.sample_data()
    assert frame.shape == (20, 5)
    assert frame.isna().any().any(), "sample data must contain missing values"
    assert not frame.isna().all().any(), "no column may be entirely missing"
    assert (frame.isna().sum() == 0).any(), "one column is meant to be complete"
    assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns)


def test_sample_data_returns_a_fresh_copy():
    first = mf.sample_data()
    first.loc[0, "age"] = 999.0
    assert mf.sample_data().loc[0, "age"] != 999.0


# sorting


def test_matrix_sort_by_a_data_column_orders_the_rows(md):
    """sort_by naming a column sorts the rows by its values, which is the MAR check:
    does missingness cluster at one end of another variable?"""
    desc = list(md.matrix(sort_by="age", ascending=False).fig.data[0].y)
    asc = list(md.matrix(sort_by="age", ascending=True).fig.data[0].y)
    assert desc != asc, "ascending is being ignored"

    observed_desc = [float(v) for v in desc if v != "NaN"]
    observed_asc = [float(v) for v in asc if v != "NaN"]
    assert observed_desc == sorted(observed_desc, reverse=True)
    assert observed_asc == sorted(observed_asc)


def test_matrix_sort_by_an_unknown_column_raises(md):
    plot = md.matrix(sort_by="not_a_column")
    with pytest.raises(ValueError, match="sort_by"):
        _ = plot.fig


@pytest.mark.parametrize(
    "name,build",
    [
        ("bar", lambda md, **k: md.bar(**k)),
        ("rate", lambda md, **k: md.rate(**k)),
        ("matrix", lambda md, **k: md.matrix(**k)),
        ("heatmap", lambda md, **k: md.heatmap(**k)),
    ],
)
def test_sort_by_reads_the_same_on_every_column_plot(md, name, build):
    """One vocabulary: sort_by names what to sort on, ascending gives the direction."""

    def columns(plot):
        fig = plot.fig
        ticks = [t for t in (fig.layout.xaxis.ticktext or []) if isinstance(t, str)]
        if ticks:
            return list(dict.fromkeys(ticks))
        drawn = []
        for trace in fig.data:
            values = getattr(trace, "x", None)
            if values is not None and len(values) and isinstance(values[0], str):
                drawn += [v for v in values if isinstance(v, str)]
        return list(dict.fromkeys(drawn))

    by_missingness = columns(build(md, sort_by="missingness", ascending=False))
    alphabetical = columns(build(md, sort_by="alphabetical", ascending=True))
    frame_order = columns(build(md, sort_by=None))

    assert by_missingness[0] == "income", f"{name}: emptiest column should lead"
    assert alphabetical == sorted(alphabetical), f"{name}: not alphabetical"
    assert frame_order == [c for c in md.columns if c in frame_order], f"{name}: not frame order"


# upset and density edge cases


def test_upset_requires_named_columns(md):
    """It used to take the top 3 by missing rate and say nothing."""
    plot = md.upset()
    with pytest.raises(ValueError, match="selected_columns"):
        _ = plot.fig


def test_upset_draws_every_column_it_is_given(md):
    """UpSet exists to scale past the three a Venn allows, so it caps nothing."""
    cols = ["age", "income", "score", "rating"]
    fig = md.upset(selected_columns=cols).fig
    assert len(set(fig.data[1].y)) == len(cols)


def test_upset_warns_rather_than_truncating_quietly():
    """Drawing the 20 largest of 40 without saying so reads as 'these are all the
    patterns', which is the opposite of what the plot is for."""
    rng = np.random.default_rng(0)
    cols = [f"c{i}" for i in range(9)]
    frame = pd.DataFrame({c: np.where(rng.random(400) < 0.3, np.nan, 1.0) for c in cols})
    plot = MissingData(frame).upset(selected_columns=cols)

    with pytest.warns(UserWarning, match="drawing the 20 largest"):
        bars = plot.fig.data[0]
    assert len(bars.y) == 20, "the cap still applies"


def test_upset_is_quiet_when_everything_fits(md):
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        _ = md.upset(selected_columns=["age", "income", "score"]).fig


def test_jitter_separates_coincident_points(md):
    """Tied values stack into one mark without it, so the plot shows one row where
    it should show many."""
    frame = pd.DataFrame({"a": [5.0] * 20 + [None] * 5, "b": [7.0] * 20 + [1.0] * 5})
    plot = MissingData(frame).scatterplot(x="a", y="b", jitter=0.0)
    drawn = plot.fig.data[0]
    assert len(set(drawn.x)) == 1, "jitter=0 draws the exact value"

    spread = MissingData(frame).scatterplot(x="a", y="b", jitter=0.05).fig.data[0]
    assert len(set(spread.x)) > 1, "jitter separates them"


def test_jitter_reaches_the_missing_offset_band(md):
    """Every missing row shares one coordinate, so the band needs the spread more
    than tied observed values do."""
    for j, expect_spread in [(0.0, False), (0.02, True)]:
        fig = md.scatterplot(x="age", y="income", jitter=j).fig
        band = [t for t in fig.data if t.name == "NA-age" and t.x is not None][0]
        assert (len(set(band.x)) > 1) is expect_spread


def test_jitter_cannot_push_missing_points_into_the_observed_range(md):
    """The offset gap is what separates 'missing' from 'small'. A large jitter used
    to be free to close it, making the two indistinguishable."""
    observed_min = md.data["age"].min()
    for j in [0.02, 0.5, 5.0]:
        fig = md.scatterplot(x="age", y="income", jitter=j).fig
        band = [t for t in fig.data if t.name == "NA-age" and t.x is not None][0]
        assert max(band.x) < observed_min, f"jitter={j} leaked into the observed range"


def test_density_handles_a_column_with_no_spread():
    """A constant group gives gaussian_kde a singular covariance; it must not crash."""
    frame = pd.DataFrame({"flat": [5.0] * 10, "other": [1.0, None] * 5})
    fig = MissingData(frame).density(column="flat", missing_column="other").fig
    assert len(fig.data) == 2
