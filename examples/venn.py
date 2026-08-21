"""The seven exclusive missingness subsets of three columns.

Each bar is one region of a three-set Venn diagram: rows missing only A, only B,
both A and B, and so on. Pick the three columns with selected_columns.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.venn(
    selected_columns=["age", "income", "score"],
    title="Gap overlap across three columns",
).show()
# mf.venn(
#     df,
#     selected_columns=["age", "income", "score"],
#     title="Gap overlap across three columns",
# ).show()

# Every option at once. Regions are ordered by size with the smallest first, and
# drawn as a share of the dataset rather than a row count, which is the comparable
# form when reporting alongside another dataset.
md.venn(
    selected_columns=["age", "income", "score"],
    sort_by="size",
    ascending=True,
    measure="percentage",
    show_values=True,
    title="Gap overlap, as a share of rows",
).show()
# mf.venn(df, selected_columns=["age", "income", "score"], sort_by="size",
#         ascending=True, measure="percentage", show_values=True,
#         title="Gap overlap, as a share of rows").show()
