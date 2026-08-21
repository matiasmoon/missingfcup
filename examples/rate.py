"""Missing rate drawn as a single coloured strip.

One row of cells, one per column, shaded by missing rate. This stays readable when
a dataset has many columns, where a bar chart would get crowded.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.rate(title="Missing rate per column").show()
# mf.rate(df, title="Missing rate per column").show()

md.rate(measure="percentage", title="Missing percentage per column").show()
# mf.rate(df, measure="percentage", title="Missing percentage per column").show()

# Every column option at once. The in-cell numbers are dropped here and the labels
# capped, which is the configuration this plot is built for: many columns, read by
# colour rather than by value, with the exact figure left to the hover.
md.rate(
    selected_columns=["age", "income", "score", "rating"],
    high_missingness_threshold=0.9,
    measure="fraction",
    show_values=False,
    max_columns=4,
    sort_by="alphabetical",
    ascending=True,
    max_label_length=8,
    title="Missing rate, read by colour",
).show()
# mf.rate(df, selected_columns=["age", "income", "score", "rating"],
#         high_missingness_threshold=0.9, measure="fraction", show_values=False,
#         max_columns=4, sort_by="alphabetical", ascending=True,
#         max_label_length=8, title="Missing rate, read by colour").show()
