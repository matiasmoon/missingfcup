"""Value-level tests: verify metrics compute the *correct* numbers, not just that
plots construct. Fixture values are hand-computed below.

Fixture (5 rows x 3 cols), missing marked with `.`:
      a    b    c
  0   1    .    1      -> 1 missing
  1   .    2    2      -> 1 missing
  2   3    3    3      -> 0 missing  (complete row)
  3   4    .    4      -> 1 missing
  4   .    5    5      -> 1 missing
     ---  ---  ---
 miss 2    2    0      total missing = 4 / 15 cells
"""

import numpy as np
import pandas as pd
import pytest

import missingfcup as mf
from missingfcup import MissingData


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "a": [1.0, None, 3.0, 4.0, None],
            "b": [None, 2.0, 3.0, None, 5.0],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def md(df):
    return MissingData(df)


def test_col_missing_count(md):
    assert md.col_missing_count["a"] == 2
    assert md.col_missing_count["b"] == 2
    assert md.col_missing_count["c"] == 0


def test_col_missing_rate(md):
    assert md.col_missing_rate["a"] == pytest.approx(0.4)
    assert md.col_missing_rate["b"] == pytest.approx(0.4)
    assert md.col_missing_rate["c"] == pytest.approx(0.0)


def test_col_present_count(md):
    assert md.col_present_count["a"] == 3
    assert md.col_present_count["c"] == 5


def test_col_completeness_is_complement(md):
    assert md.col_completeness["a"] == pytest.approx(0.6)
    assert md.col_completeness["c"] == pytest.approx(1.0)


def test_col_missing_percent(md):
    assert md.col_missing_percent["a"] == pytest.approx(40.0)
    assert md.col_missing_percent["c"] == pytest.approx(0.0)


def test_cols_complete(md):
    assert set(md.cols_complete) == {"c"}


def test_row_missing_count(md):
    assert list(md.row_missing_count) == [1, 1, 0, 1, 1]


def test_rows_complete(md):
    assert set(md.rows_complete) == {2}


def test_total_missing_count(md):
    assert md.total_missing_count == 4


def test_total_missing_rate(md):
    assert md.total_missing_rate == pytest.approx(4 / 15)


def test_mask_missing_values(md):
    assert md.mask_missing.shape == (5, 3)
    assert md.mask_missing.loc[0, "a"] == False  # noqa: E712 present
    assert md.mask_missing.loc[0, "b"] == True  # noqa: E712 missing
    assert not md.mask_missing.loc[2].any()  # row 2 fully present


def test_mask_present_is_inverse(md):
    assert (md.mask_present == ~md.mask_missing).all().all()


def test_totals_plot_data(md):
    """totals() must encode 11 present vs 4 missing cells."""
    fig = md.totals().fig
    bar = fig.data[0]
    assert list(bar.x) == ["!NA", "NA"]
    assert tuple(bar.y) == (11, 4)


def test_rejects_duplicate_columns():
    df = pd.DataFrame([[1, None, 3], [None, 2, 3]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="duplicate column"):
        MissingData(df)


# Error contract: ValueError when the call cannot produce a meaningful result,
# TypeError for a wrong dtype or input type, and every message names the offending
# value and says what would change it.


def test_a_missing_column_raises_the_same_type_everywhere():
    """It used to be KeyError from the metrics and ValueError from the plots, so a
    caller could not write one except clause for the same mistake."""
    md = MissingData(mf.sample_data())
    calls = [
        lambda: md.mann_whitney_test(x="nope", by="age"),
        lambda: md.mann_whitney_test(x="age", by="nope"),
        lambda: md.boxplot(column="nope", missing_column="age").fig,
        lambda: md.density(column="nope", missing_column="age").fig,
        lambda: md.scatterplot(x="nope", y="age").fig,
    ]
    for call in calls:
        with pytest.raises(ValueError, match="nope"):
            call()


@pytest.mark.parametrize("threshold", [-0.5, 1.5])
def test_threshold_errors_report_the_value_given(threshold):
    md = MissingData(mf.sample_data())
    for call in (md.columns_above_missing_threshold, md.rows_above_missing_threshold):
        with pytest.raises(ValueError, match=repr(threshold)):
            call(threshold)


def test_littles_test_error_says_how_to_proceed():
    """A dead-end message names the failure; a useful one names the way out."""
    frame = pd.DataFrame({"a": ["x", "y", None], "b": ["p", None, "q"]})
    with pytest.raises(ValueError, match="numeric_only=False"):
        MissingData(frame).littles_mcar_test()


def test_value_missing_corr_nan_contract():
    """The matrix is built a row at a time rather than a cell at a time, so the cases
    that produce NaN have to be pinned: a non-numeric column, a constant column, a
    fully missing column, and a missingness column whose mask never varies."""
    frame = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
            "text": list("abcde"),
            "constant": [7.0] * 5,
            "all_nan": [None] * 5,
            "gappy": [1.0, None, None, 4.0, 5.0],
        }
    )
    corr = MissingData(frame).value_missing_corr

    assert corr.loc["text"].isna().all(), "non-numeric values cannot be correlated"
    assert corr.loc["constant"].isna().all(), "a constant has no variance"
    assert corr.loc["all_nan"].isna().all(), "no observed values at all"
    assert corr["numeric"].isna().all(), "a column that is never missing has no pattern"
    assert not np.isnan(corr.loc["numeric", "gappy"]), "the one computable cell"


def test_value_missing_corr_matches_a_pairwise_pearson():
    """Equivalence with the per-pair definition it is a vectorised form of."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(200, 6))).add_prefix("c")
    frame = frame.mask(rng.random((200, 6)) < 0.3)
    md = MissingData(frame)

    for value_col in frame.columns:
        x = frame[value_col]
        for missing_col in frame.columns:
            observed = x.notna()
            expected = x[observed].corr(md.mask_missing[missing_col][observed].astype(float))
            actual = md.value_missing_corr.loc[value_col, missing_col]
            assert np.isclose(actual, expected, equal_nan=True)
