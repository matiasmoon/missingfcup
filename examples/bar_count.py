"""Per-column missing counts.

The default bar chart. Use show_both=True to stack present against missing, so the
column height becomes the row count and the split is visible at a glance.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.bar(title="Missing values per column").show()
# mf.bar(df, title="Missing values per column").show()

# Present and missing stacked together.
md.bar(
    show_both=True,
    text_color="#FFFFFF",
    background_color="#326556",
    title="How much of each column is filled",
).show()
# mf.bar(df, show_both=True, title="How much of each column is filled").show()

# Every column option at once. ascending=True orders by missing rate with the
# cleanest first, so capping at three keeps the three *least* affected columns --
# flip ascending to keep the emptiest instead. Horizontal bars keep long column
# names readable, since they run along the axis rather than rotated under it.
md.bar(
    measure="count",
    selected_columns=["age", "income", "score", "rating"],
    high_missingness_threshold=0.9,
    max_columns=3,
    sort_by="missingness",
    ascending=True,
    orientation="horizontal",
    show_values=False,
    title="The three columns with fewest gaps",
).show()
# mf.bar(df, measure="count", selected_columns=["age", "income", "score", "rating"],
#        high_missingness_threshold=0.9, max_columns=3, sort_by="missingness",
#        ascending=True, orientation="horizontal", show_values=False,
#        title="The three columns with fewest gaps").show()
