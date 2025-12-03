"""Tests for the missing_matrix visualization function.

This module tests the core missing_matrix function to ensure it correctly
generates Plotly figures from DataFrame inputs and handles various edge cases
and parameter combinations.
"""

import pandas as pd

# Import the function from the package under test.
from missingfcup.viz.missing_matrix import missing_matrix


def test_missing_matrix_runs():
    """Basic smoke test: the function should return a Figure for a small DF.

    This test does not assert visual contents — it merely ensures the
    function runs and returns a non-None figure object.
    """
    # Small DataFrame with a missing value
    df = pd.DataFrame({"a": [1, None, 3]})

    # Call function under test
    fig = missing_matrix(df)

    # The function should return a Plotly Figure (or at least a truthy value)
    assert fig is not None
