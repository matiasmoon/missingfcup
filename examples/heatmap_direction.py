"""Biserial heatmap: do a column's *values* relate to another column's gaps?

Point-biserial correlation between the observed values of one column and the
missingness indicator of another. Unlike the other two heatmaps, this reads the
actual numbers, not just the missingness pattern, so it needs numeric columns.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.heatmap(kind="direction", title="Do values in one column predict gaps in another").show()
# mf.heatmap(df, kind="direction", title="Do values in one column predict gaps in another").show()

# This is the one heatmap whose axes mean different things, so each gets its own
# selection: rows are the columns whose values are read, columns are the ones whose
# missingness is tested. Nothing here mirrors anything, which is why
# show_upper_triangle is refused for this kind.
md.heatmap(
    kind="direction",
    selected_value_columns=["age", "income"],
    selected_missing_columns=["score", "rating"],
    title="Do age and income predict gaps in score and rating",
).show()
# mf.heatmap(df, kind="direction", selected_value_columns=["age", "income"],
#            selected_missing_columns=["score", "rating"],
#            title="Do age and income predict gaps in score and rating").show()
