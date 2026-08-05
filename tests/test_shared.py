"""Tests for the shared plot helpers: column selection and label truncation.

Fixture columns and their missing rates:
    a, b -> 0.4 (two of five missing)
    c    -> 0.0 (complete)
    d    -> 1.0 (fully missing)
"""
import pytest
import pandas as pd

from missingfcup import MissingData
from missingfcup.plots._selection import select_columns


@pytest.fixture
def md():
    return MissingData(pd.DataFrame({
        "a": [1.0, None, 3.0, 4.0, None],
        "b": [None, 2.0, 3.0, None, 5.0],
        "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        "d": [None, None, None, None, None],
    }))


# ---------------------------------------------------------------------------
# select_columns
# ---------------------------------------------------------------------------

def test_select_all_by_default(md):
    assert select_columns(md) == ["a", "b", "c", "d"]


def test_ignore_high_missingness_drops_near_empty(md):
    cols = select_columns(md, ignore_high_missingness=True, high_missingness_threshold=0.9)
    assert set(cols) == {"a", "b", "c"}   # d (fully missing) dropped


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


def test_order_by_missingness_desc(md):
    cols = select_columns(md, order_by_missingness=True, order="desc")
    assert cols[0] == "d"    # fully missing first
    assert cols[-1] == "c"   # complete last


def test_order_by_missingness_asc(md):
    cols = select_columns(md, order_by_missingness=True, order="asc")
    assert cols[0] == "c"
    assert cols[-1] == "d"


# ---------------------------------------------------------------------------
# _Plot._truncate_labels
# ---------------------------------------------------------------------------

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
    assert len(set(out)) == len(out)   # the duplicate the cut created is disambiguated
