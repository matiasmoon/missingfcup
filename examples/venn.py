"""The seven exclusive missingness subsets of three columns.

Each bar is one region of a three-set Venn diagram: rows missing only A, only B,
both A and B, and so on. Pick the three columns with selected_columns.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.venn(
    selected_columns=["age", "income", "score"],
    title="Missingness overlap across three columns",
).show()
