"""Tests for the shared plot helpers: column selection and label truncation.

Fixture columns and their missing rates:
    a, b -> 0.4 (two of five missing)
    c    -> 0.0 (complete)
    d    -> 1.0 (fully missing)
"""

import pandas as pd
import pytest

from missingfcup import MissingData
from missingfcup.plots._selection import select_columns


@pytest.fixture
def md():
    return MissingData(
        pd.DataFrame(
            {
                "a": [1.0, None, 3.0, 4.0, None],
                "b": [None, 2.0, 3.0, None, 5.0],
                "c": [1.0, 2.0, 3.0, 4.0, 5.0],
                "d": [None, None, None, None, None],
            }
        )
    )


def test_select_all_by_default(md):
    assert select_columns(md) == ["a", "b", "c", "d"]


def test_high_missingness_threshold_drops_near_empty(md):
    assert set(select_columns(md, high_missingness_threshold=0.9)) == {"a", "b", "c"}  # d dropped
    assert set(select_columns(md)) == {"a", "b", "c", "d"}  # None keeps everything


def test_selected_columns_kept_in_given_order(md):
    assert select_columns(md, ["b", "a"]) == ["b", "a"]


def test_selected_columns_none_match_raises(md):
    with pytest.raises(ValueError, match="selected_columns"):
        select_columns(md, ["nope"])


def test_max_columns_caps(md):
    assert select_columns(md, max_columns=2) == ["a", "b"]


def test_drop_constant_removes_constant_missingness(md):
    # c is always present and d always missing, so both have a constant mask
    assert set(select_columns(md, drop_constant=True)) == {"a", "b"}


def test_sort_by_missingness_desc(md):
    cols = select_columns(md, sort_by="missingness", ascending=False)
    assert cols[0] == "d"  # fully missing first
    assert cols[-1] == "c"  # complete last


def test_sort_by_missingness_asc(md):
    cols = select_columns(md, sort_by="missingness", ascending=True)
    assert cols[0] == "c"
    assert cols[-1] == "d"


def test_sort_by_none_keeps_the_dataframe_order(md):
    """sort_by names what to sort on; None leaves the frame's own order alone."""
    assert select_columns(md, sort_by=None) == ["a", "b", "c", "d"]


def test_sort_by_alphabetical(md):
    assert select_columns(md, sort_by="alphabetical", ascending=True) == ["a", "b", "c", "d"]
    assert select_columns(md, sort_by="alphabetical", ascending=False) == ["d", "c", "b", "a"]


def test_short_labels_unchanged(md):
    p = md.matrix()
    p.max_label_length = 10
    assert p._truncate_labels(["ab", "cd"]) == ["ab", "cd"]


def test_long_label_truncated(md):
    p = md.matrix()
    p.max_label_length = 5
    assert p._truncate_labels(["abcdefghij"]) == ["abcd…"]


def test_truncation_collisions_made_unique(md):
    p = md.matrix()
    p.max_label_length = 5
    out = p._truncate_labels(["abcdefX", "abcdefY"])
    assert len(set(out)) == len(out)  # the duplicate the cut created is disambiguated


def test_truncation_collisions_stay_within_the_cap(md):
    """Disambiguating must not push a label past max_label_length: the padding is
    invisible on screen but the cap is still the cap."""
    p = md.matrix()
    p.max_label_length = 6
    out = p._truncate_labels(["abcdefgh"] * 4)
    assert len(set(out)) == len(out)
    assert max(len(label) for label in out) <= 6, out


@pytest.mark.parametrize(
    "name,build",
    [
        ("matrix", lambda md, n: md.matrix(max_label_length=n)),
        ("rate", lambda md, n: md.rate(max_label_length=n)),
        ("upset", lambda md, n: md.upset(selected_columns=list(md.columns), max_label_length=n)),
        ("bar", lambda md, n: md.bar(max_label_length=n)),
        ("bar_rate", lambda md, n: md.bar(measure="fraction", max_label_length=n)),
        ("heatmap", lambda md, n: md.heatmap(max_label_length=n)),
        ("dendrogram", lambda md, n: md.dendrogram(max_label_length=n)),
        ("parallel_coordinates", lambda md, n: md.parallel_coordinates(max_label_length=n)),
    ],
)
def test_max_label_length_truncates_every_plot_that_shows_column_names(name, build):
    """Any plot putting column names on an axis has to honour the cap, or a wide
    dataset pushes the drawing off the figure."""
    long_names = [f"{letter}_{'x' * 40}" for letter in "abcd"]
    frame = pd.DataFrame({c: [1.0, None, 3.0, None, 5.0] for c in long_names})
    frame.iloc[0, 1] = 2.0  # give the columns differing missingness
    plot = build(MissingData(frame), 12)

    drawn = []
    for trace in plot.fig.data:
        for axis in ("x", "y"):
            values = getattr(trace, axis, None)
            if values is not None and len(values) and isinstance(values[0], str):
                drawn += list(values)
    ticks = getattr(plot.fig.layout.xaxis, "ticktext", None)
    if ticks:
        drawn += list(ticks)

    # upset mixes None into its category arrays as spacers.
    drawn = [label for label in drawn if isinstance(label, str)]
    assert drawn, f"{name} drew no text labels"
    assert max(len(label) for label in drawn) <= 12, (
        f"{name} drew a label longer than max_label_length"
    )


def test_panel_save_writes_the_panel_and_each_plot(md, tmp_path):
    from missingfcup import Panel

    out = tmp_path / "panel.html"
    Panel([md.matrix(), md.bar()]).save(str(out), save_individual=True)

    assert out.exists()
    individual = sorted(p.name for p in tmp_path.glob("*.png"))
    assert len(individual) == 2, f"expected one PNG per plot, got {individual}"


# Hover text: one grammar across every plot, in the same words as the legends.


def _templates(plot):
    """Every hovertemplate in a figure, skipping traces that opt out deliberately."""
    return [
        t.hovertemplate
        for t in plot.fig.data
        if getattr(t, "hoverinfo", None) != "skip" and getattr(t, "hovertemplate", None)
    ]


def _all_plots(md):
    return {
        "matrix": md.matrix(),
        "bar": md.bar(),
        "bar_both": md.bar(show_both=True),
        "bar_rate": md.bar(measure="fraction"),
        "rate": md.rate(),
        "totals": md.totals(),
        "heatmap": md.heatmap(),
        "heatmap_direction": md.heatmap(kind="direction"),
        "heatmap_dependence": md.heatmap(kind="dependence"),
        "dendrogram": md.dendrogram(),
        "venn": md.venn(selected_columns=["a", "b", "c"]),
        "upset": md.upset(selected_columns=["a", "b", "c"]),
        "scatterplot": md.scatterplot(x="a", y="b"),
        "density": md.density(column="a", missing_column="b"),
        "boxplot": md.boxplot(column="a", missing_column="b"),
        # d is all-None, so it has no dtype to normalise onto a shared axis.
        "parallel_coordinates": md.parallel_coordinates(
            selected_columns=["a", "b", "c"], missing_column="a"
        ),
    }


def test_every_trace_designs_its_own_hover(md):
    """A trace with no template falls back to plotly's raw x/y dump, which ignores
    the package's vocabulary and its number formats."""
    for name, plot in _all_plots(md).items():
        for trace in plot.fig.data:
            skipped = getattr(trace, "hoverinfo", None) == "skip"
            assert skipped or getattr(trace, "hovertemplate", None), (
                f"{name} has a trace with no hover template"
            )


def test_no_hover_uses_the_retired_vocabulary(md):
    """The legends say NA and !NA. A tooltip saying 'Present' would give the same
    point two different names in one figure."""
    retired = ["Status", "Present", "Percent:", "Count:", "Axes"]
    for name, plot in _all_plots(md).items():
        for template in _templates(plot):
            for word in retired:
                assert word not in template, f"{name} still says {word!r}: {template}"


def test_hover_suppresses_the_trace_name_chip(md):
    """Plotly appends a trace-name box unless the template ends it. The name is
    already in the legend, so the box only widens the tooltip."""
    for name, plot in _all_plots(md).items():
        for template in _templates(plot):
            assert template.endswith("<extra></extra>"), f"{name}: {template}"


def test_counts_are_shown_against_their_total(md):
    """A bare count cannot be judged without knowing the size of the dataset."""
    for name in ["bar", "bar_rate", "rate", "venn", "upset", "density", "boxplot"]:
        plot = _all_plots(md)[name]
        # The total reaches the tooltip either through customdata or baked into
        # the template, depending on whether it varies per point.
        rendered = " ".join(
            [t.hovertemplate or "" for t in plot.fig.data]
            + [
                str(v)
                for trace in plot.fig.data
                for row in (trace.customdata if trace.customdata is not None else [])
                for v in (row if isinstance(row, (list, tuple)) else [row])
            ]
        )
        assert "of 5 rows" in rendered, f"{name} shows a count without its total"
