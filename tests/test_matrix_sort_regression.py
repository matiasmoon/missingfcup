"""Regression: sort_by must not collapse rows onto shared y coordinates.

A sort column repeats its values. When those values were passed to go.Heatmap as the
y coordinate, every row sharing a value landed on the same row of the figure and
overplotted, so the missing cells vanished from the rendered matrix while the z data
stayed correct. See plots/_matrix.py.
"""

import pandas as pd
import pytest

from missingfcup import MissingData


@pytest.fixture
def repeated_driver():
    # 60 rows, a driver taking only three distinct values, gaps in one column.
    return pd.DataFrame(
        {
            "driver": [1] * 20 + [2] * 20 + [3] * 20,
            "target": [None] * 15 + list(range(45)),
        }
    )


def test_sort_by_keeps_every_row_distinct(repeated_driver):
    trace = MissingData(repeated_driver).matrix(sort_by="driver").fig.data[0]
    assert len(set(map(str, trace.y))) == len(trace.y)


def test_sort_by_preserves_missing_cells(repeated_driver):
    trace = MissingData(repeated_driver).matrix(sort_by="driver").fig.data[0]
    zeros = sum(1 for row in trace.z for cell in row if cell == 0)
    assert zeros == repeated_driver.isna().sum().sum()
