"""Parallel coordinates coloured by one column's missingness.

Every row becomes a line crossing all the axes. Lines are coloured by whether the
chosen column is missing in that row, so a multivariate pattern behind the gaps
shows up as the two colours separating.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.parallel_coordinates(
    missing_column="age",
    title="Every row, coloured by gaps in age",
).show()
# mf.parallel_coordinates(
#     df,
#     missing_column="age",
#     title="Every row, coloured by gaps in age",
# ).show()

# Every option at once. Axis order matters more here than on any other plot: a
# relationship shows up only between neighbouring axes, so ordering by missing rate
# puts the columns most likely to share gaps side by side.
md.parallel_coordinates(
    selected_columns=["age", "income", "score", "rating"],
    missing_column="age",
    high_missingness_threshold=0.9,
    max_columns=4,
    sort_by="missingness",
    ascending=False,
    kind="values",
    title="Emptiest columns placed side by side",
).show()
# mf.parallel_coordinates(df, selected_columns=["age", "income", "score", "rating"],
#                         missing_column="age", high_missingness_threshold=0.9,
#                         max_columns=4, sort_by="missingness", ascending=False,
#                         kind="values", title="Emptiest columns placed side by side").show()

# kind="missingness" draws every column as present/missing instead of by value. It
# is the escape hatch for non-numeric columns, which cannot be normalised onto a
# shared axis at all.
md.parallel_coordinates(
    missing_column="age",
    kind="missingness",
    title="Every column as present or missing",
).show()
# mf.parallel_coordinates(df, missing_column="age", kind="missingness",
#                         title="Every column as present or missing").show()
