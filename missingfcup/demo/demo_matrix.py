"""Demo: Basic missing_matrix visualization example.

This script constructs a small example DataFrame and renders the
interactive missingness matrix. It is intended to be a lightweight
manual demo you can run locally (e.g. `python -m missingfcup.demo.demo_matrix`).

It demonstrates the basic usage of the missing_matrix function with
default settings and optional customization.
"""

import pandas as pd

# Import the primary visualization function from the package.
from missingfcup.viz.missing_matrix import missing_matrix


# Create a tiny example DataFrame with some missing values.
df = pd.DataFrame({
    "A": [1, None, 3, 4, None],  # numeric with missing entries
    "B": ["x", "y", None, "z", "w"],  # categorical with a missing
    "C": [10, 20, 30, None, 50],  # numeric with one missing value
})


# Build a figure using the default settings. This returns a Plotly Figure object.
fig = missing_matrix(df)  # super simple: defaults are used for colors, sizing

# Display the figure in an interactive window (Plotly opens in browser / renderer).
fig.show()


# Example of a more customized call (left commented):
# fig = missing_matrix(
#     df,
#     max_cols=100,
#     height=800,
#     width=1200,
#     present_color="#1f77b4",
#     missing_color="#ff7f0e",
#     selected_columns=["A", "C"],
#     show_hover=True,
#     show_scale=True
# )
# fig.show()
