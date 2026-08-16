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
    ("bar_rate", lambda md: md.bar(measure="rate")),
    ("rate", lambda md: md.rate()),
    ("totals", lambda md: md.totals()),
    ("heatmap_correlation", lambda md: md.heatmap()),
    ("heatmap_predictive", lambda md: md.heatmap(kind="predictive")),
    ("heatmap_biserial", lambda md: md.heatmap(kind="biserial")),
    ("dendrogram", lambda md: md.dendrogram()),
    ("venn", lambda md: md.venn(selected_columns=["age", "income", "score"])),
    ("upset", lambda md: md.upset()),
    ("scatterplot", lambda md: md.scatterplot(x="age", y="income")),
    ("density", lambda md: md.density(x="income", color_by="age")),
    ("boxplot", lambda md: md.boxplot(x="income", color_by="age")),
    ("parallel_coordinates", lambda md: md.parallel_coordinates(missingness_color_column="age")),
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
    z = np.asarray(md.matrix(ignore_high_missingness=False).fig.data[0].z)
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


def test_bar_rate_percentage_is_fraction_times_100(md):
    fraction = md.bar(measure="rate").fig.data[0].y
    percentage = md.bar(measure="rate", scale="percentage").fig.data[0].y
    assert np.allclose(np.asarray(percentage), np.asarray(fraction) * 100)


def test_rate_strip_is_one_row_of_missing_rates(md):
    z = np.asarray(md.rate().fig.data[0].z)
    assert z.shape[0] == 1, "rate() draws a single row"
    assert ((z >= 0) & (z <= 1)).all()


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
    """A never-missing column has no missingness variance, so it reads as NaN."""
    heat = md.heatmap().fig.data[-1]
    z = np.asarray(heat.z, dtype=float)
    visits = list(heat.x).index("visits")  # the complete column in sample_data
    assert np.isnan(z[visits][visits])


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
    fig = md.upset(selected_columns=cols, max_sets=3).fig
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
    fig = md.density(x="income", color_by="age").fig
    assert len(fig.data) == 2
    assert {trace.name for trace in fig.data} == {"NA", "!NA"}


def test_boxplot_splits_rows_by_missingness_of_the_other_column(md, df):
    fig = md.boxplot(x="income", color_by="age").fig
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


# Options that change the output


def test_selected_columns_restricts_what_is_drawn(md):
    bar = md.bar(selected_columns=["age", "income"]).fig.data[0]
    assert list(bar.x) == ["age", "income"]


def test_order_by_missingness_sorts_the_rate_strip(md):
    """Ordering acts on the values, not the labels.

    Tied columns keep their original relative order in both directions (the sort is
    stable), so asc is not simply desc reversed. Monotonicity is the real contract.
    """
    desc = np.asarray(md.rate(order="desc").fig.data[0].z)[0]
    asc = np.asarray(md.rate(order="asc").fig.data[0].z)[0]
    assert (np.diff(desc) <= 0).all(), "desc must not increase"
    assert (np.diff(asc) >= 0).all(), "asc must not decrease"
    assert sorted(desc) == sorted(asc), "same columns, different order"


def test_heatmap_text_font_size_reaches_the_trace(md):
    assert md.heatmap(text_font_size=18).fig.data[-1].textfont.size == 18


def test_semantic_colors_are_configurable(md):
    fig = md.bar(missing_color="#123456").fig
    assert fig.data[0].marker.color == "#123456"


COLUMN_AXIS_PLOTS = [
    ("bar", lambda md: md.bar(), "Column"),
    ("bar_rate", lambda md: md.bar(measure="rate"), "Column"),
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
    assert fig.layout.xaxis.title.text == "Count"


# Errors


@pytest.mark.parametrize(
    "call,expected",
    [
        (lambda md: md.bar(measure="nope"), ValueError),
        (lambda md: md.heatmap(kind="nope"), ValueError),
        (lambda md: md.heatmap(kind="correlation", selected_value_columns=["age"]), ValueError),
        (lambda md: md.scatterplot(x="age", y="absent").fig, ValueError),
        (lambda md: md.boxplot(x="absent", color_by="age").fig, ValueError),
        (lambda md: md.density(x="absent", color_by="age").fig, ValueError),
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
    plot = MissingData(frame).boxplot(x="label", color_by="age")
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
    flat = mf.bar(df, measure="rate").fig.data[0]
    method = MissingData(df).bar(measure="rate").fig.data[0]
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


# order_by validation


def test_order_by_direction_alias_is_honoured(md):
    """`direction` used to be silently ignored, so desc rendered identically to asc.

    Missing values sort last in both directions, so only the observed rows reverse.
    """
    desc = [v for v in md.matrix(order_by=[{"column": "age", "direction": "desc"}]).fig.data[0].y]
    asc = [v for v in md.matrix(order_by=[{"column": "age", "direction": "asc"}]).fig.data[0].y]
    assert desc != asc, "direction is being ignored"

    observed_desc = [float(v) for v in desc if v != "NaN"]
    observed_asc = [float(v) for v in asc if v != "NaN"]
    assert observed_desc == sorted(observed_desc, reverse=True)
    assert observed_asc == sorted(observed_asc)
    assert observed_desc == list(reversed(observed_asc))


def test_order_by_direction_matches_ascending(md):
    """The alias and the canonical key must produce identical output."""
    via_alias = md.matrix(order_by=[{"column": "age", "direction": "desc"}]).fig.data[0].y
    via_ascending = md.matrix(order_by=[{"column": "age", "ascending": False}]).fig.data[0].y
    assert list(via_alias) == list(via_ascending)


@pytest.mark.parametrize(
    "spec,message",
    [
        ({"column": "age", "typo": 1}, "unknown key"),
        ({"axis": "rows"}, "must have a 'column'"),
        ({"column": "age", "axis": "sideways"}, "axis="),
        ({"column": "age", "direction": "up"}, "direction="),
        ({"column": "age", "type": "ordinal"}, "type="),
        ({"column": "age", "ascending": True, "direction": "desc"}, "contradict"),
    ],
)
def test_order_by_rejects_bad_specs(md, spec, message):
    """An unrecognised or contradictory spec must fail loudly, not be ignored."""
    with pytest.raises(ValueError, match=message):
        md.matrix(order_by=[spec])._build_figure()


def test_bar_accepts_the_same_order_by_dialect_as_matrix(md):
    """bar() used to raise KeyError on a spec matrix() accepted."""
    fig = md.bar(order_by=[{"column": "__missing__", "ascending": False}]).fig
    counts = list(fig.data[0].y)
    assert counts == sorted(counts, reverse=True)


def test_bar_rejects_ordering_that_could_not_affect_it(md):
    """A bar shows one bar per column, so sorting rows cannot change it."""
    with pytest.raises(ValueError, match="order of rows"):
        md.bar(order_by=[{"column": "age"}])._build_figure()


# upset and density edge cases


def test_upset_warns_instead_of_silently_dropping_columns(md):
    with pytest.warns(UserWarning, match="dropped"):
        md.upset(selected_columns=["age", "income", "score", "rating"])._build_figure()


def test_upset_respects_a_raised_max_sets(md):
    fig = md.upset(selected_columns=["age", "income", "score", "rating"], max_sets=4).fig
    assert len(set(fig.data[1].y)) == 4


def test_upset_highlight_without_an_explicit_colour(md):
    """highlight_columns alone used to put None into the colour list and crash."""
    fig = md.upset(selected_columns=["age", "income"], highlight_columns=["age"]).fig
    assert all(colour is not None for colour in fig.data[1].marker.color)


def test_density_handles_a_column_with_no_spread():
    """A constant group gives gaussian_kde a singular covariance; it must not crash."""
    frame = pd.DataFrame({"flat": [5.0] * 10, "other": [1.0, None] * 5})
    fig = MissingData(frame).density(x="flat", color_by="other").fig
    assert len(fig.data) == 2
