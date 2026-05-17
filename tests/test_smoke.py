"""Smoke tests: verify the package imports and core API is reachable."""
import pytest
import pandas as pd

from missingfcup import MissingData, Panel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "a": [1.0, None, 3.0, 4.0, None],
        "b": [None, 2.0, 3.0, None, 5.0],
        "c": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def md(simple_df):
    return MissingData(simple_df)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction(simple_df):
    assert MissingData(simple_df) is not None


def test_rejects_non_dataframe():
    with pytest.raises(TypeError):
        MissingData([[1, 2], [3, 4]])


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        MissingData(pd.DataFrame())


# ---------------------------------------------------------------------------
# Core masks
# ---------------------------------------------------------------------------

def test_mask_missing_shape(md, simple_df):
    assert md.mask_missing.shape == simple_df.shape


def test_mask_present_is_inverse(md):
    assert (md.mask_missing == ~md.mask_present).all().all()


# ---------------------------------------------------------------------------
# Column metrics
# ---------------------------------------------------------------------------

def test_col_missing_rate_range(md):
    assert (md.col_missing_rate >= 0).all()
    assert (md.col_missing_rate <= 1).all()


def test_col_completeness_complement(md):
    diff = (md.col_missing_rate + md.col_completeness - 1.0).abs()
    assert (diff < 1e-10).all()


def test_cols_complete_has_no_missing(md):
    for col in md.cols_complete:
        assert md.col_missing_count[col] == 0


# ---------------------------------------------------------------------------
# Row metrics
# ---------------------------------------------------------------------------

def test_row_missing_rate_range(md):
    assert (md.row_missing_rate >= 0).all()
    assert (md.row_missing_rate <= 1).all()


def test_rows_complete_have_no_missing(md):
    for idx in md.rows_complete:
        assert md.row_missing_count[idx] == 0


def test_rows_above_threshold_invalid(md):
    with pytest.raises(ValueError):
        md.rows_above_missing_threshold(1.5)


# ---------------------------------------------------------------------------
# Dataset totals
# ---------------------------------------------------------------------------

def test_total_missing_rate_range(md):
    assert 0.0 <= md.total_missing_rate <= 1.0


# ---------------------------------------------------------------------------
# Pattern analysis
# ---------------------------------------------------------------------------

def test_missing_pattern_in_rows_length(md, simple_df):
    assert len(md.missing_pattern_in_rows) == len(simple_df)


def test_missing_pattern_counts_max_patterns(md):
    counts = md.missing_pattern_counts(max_patterns=2)
    assert len(counts) <= 2


# ---------------------------------------------------------------------------
# Plots — verify they construct without error
# ---------------------------------------------------------------------------

def test_heatmap_builds(md):
    assert md.heatmap().fig is not None


def test_barchart_count_builds(md):
    assert md.barchart_count().fig is not None


def test_barchart_rate_builds(md):
    assert md.barchart_rate().fig is not None


def test_barchart_total_builds(md):
    assert md.barchart_total().fig is not None


def test_heatmap_rate_builds(md):
    assert md.heatmap_rate().fig is not None


def test_heatmap_correlation_builds(md):
    assert md.heatmap_correlation().fig is not None


def test_heatmap_predictive_builds(md):
    assert md.heatmap_predictive().fig is not None


def test_scatterplot_builds(md):
    assert md.scatterplot(x="a", y="b").fig is not None


def test_boxplot_builds(md):
    assert md.boxplot(x="a", color_by="b").fig is not None


def test_density_builds(md):
    assert md.density(x="a", color_by="b").fig is not None


def test_parallel_coordinates_builds(md):
    assert md.parallel_coordinates(selected_columns=["a", "b", "c"]).fig is not None


def test_dendrogram_builds(md):
    assert md.dendrogram().fig is not None


def test_barchart_venn_builds(md):
    assert md.barchart_venn(selected_columns=["a", "b", "c"]).fig is not None


def test_barchart_intersection_builds(md):
    assert md.barchart_intersection(selected_columns=["a", "b", "c"]).fig is not None


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def test_panel_builds(md):
    panel = Panel([md.heatmap(), md.barchart_count()])
    assert panel._create_combined_figure() is not None


def test_panel_add(md):
    panel = Panel()
    panel.add(md.heatmap()).add(md.barchart_count())
    assert len(panel.plots) == 2


def test_panel_empty_raises():
    with pytest.raises(ValueError):
        Panel()._create_combined_figure()


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

def test_version_exposed():
    import missingfcup
    assert hasattr(missingfcup, "__version__")
    assert isinstance(missingfcup.__version__, str)
