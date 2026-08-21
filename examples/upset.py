"""UpSet plot of every missingness combination.

The Venn approach stops working past three columns; UpSet scales. Bars give the
size of each intersection, and the dot matrix underneath says which columns that
intersection covers.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.upset(
    selected_columns=["age", "income", "score", "rating"],
    title="Which columns are missing together",
).show()
# mf.upset(
#     df,
#     selected_columns=["age", "income", "score", "rating"],
#     title="Which columns are missing together",
# ).show()

# Every option at once, and the same vocabulary venn() uses: measure scales both
# bar panels together, so the intersection sizes and the per-column totals stay
# comparable against each other.
md.upset(
    selected_columns=["age", "income", "score", "rating"],
    sort_by="size",
    ascending=True,
    measure="fraction",
    show_values=True,
    title="Which columns are missing together, as a share of rows",
).show()
# mf.upset(df, selected_columns=["age", "income", "score", "rating"],
#          sort_by="size", ascending=True, measure="fraction", show_values=True,
#          title="Which columns are missing together, as a share of rows").show()
