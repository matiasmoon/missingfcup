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

import pandas as pd
import pytest

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


# ---------------------------------------------------------------------------
# Column metrics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Row metrics
# ---------------------------------------------------------------------------


def test_row_missing_count(md):
    assert list(md.row_missing_count) == [1, 1, 0, 1, 1]


def test_rows_complete(md):
    assert set(md.rows_complete) == {2}


# ---------------------------------------------------------------------------
# Dataset totals
# ---------------------------------------------------------------------------


def test_total_missing_count(md):
    assert md.total_missing_count == 4


def test_total_missing_rate(md):
    assert md.total_missing_rate == pytest.approx(4 / 15)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def test_mask_missing_values(md):
    assert md.mask_missing.shape == (5, 3)
    assert md.mask_missing.loc[0, "a"] == False  # noqa: E712 present
    assert md.mask_missing.loc[0, "b"] == True  # noqa: E712 missing
    assert not md.mask_missing.loc[2].any()  # row 2 fully present


def test_mask_present_is_inverse(md):
    assert (md.mask_present == ~md.mask_missing).all().all()


# ---------------------------------------------------------------------------
# Plot output correctness (not just "it built")
# ---------------------------------------------------------------------------


def test_totals_plot_data(md):
    """totals() must encode 11 present vs 4 missing cells."""
    fig = md.totals().fig
    bar = fig.data[0]
    assert list(bar.x) == ["Present", "Missing"]
    assert tuple(bar.y) == (11, 4)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_duplicate_columns():
    df = pd.DataFrame([[1, None, 3], [None, 2, 3]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="duplicate column"):
        MissingData(df)
