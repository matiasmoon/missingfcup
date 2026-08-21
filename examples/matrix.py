"""Missingness matrix: one row per observation, one column per variable.

Green cells are present, red cells are missing. This is the fastest way to see
whether the gaps cluster in particular rows or columns, or fall evenly.

Commented are the flat functions that produce the same plot as object calls.
"""

import pandas as pd

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.matrix(title="Where the gaps are").show()
# mf.matrix(df, title="Where the gaps are").show()

# Every column option at once: keep four columns, drop any that is 90% empty, cap
# the count, and order what is left by missing rate with the emptiest first.
md.matrix(
    selected_columns=["age", "income", "score", "rating"],
    high_missingness_threshold=0.9,
    max_columns=4,
    sort_by="missingness",
    ascending=False,
    width=1000,
    height=600,
    title="Four columns, emptiest first",
).show()
# mf.matrix(df, selected_columns=["age", "income", "score", "rating"],
#           high_missingness_threshold=0.9, max_columns=4,
#           sort_by="missingness", ascending=False,
#           width=1000, height=600, title="Four columns, emptiest first").show()

# Naming a column in sort_by orders the *rows* by that column's values instead, and
# labels the y-axis with them. A nominal categorical has no inherent first or last,
# so sort_categories is how the order gets decided rather than falling to alphabetical.
banded = df.assign(
    band=pd.cut(df["age"], bins=[0, 30, 45, 100], labels=["young", "middle", "senior"])
)
mf.MissingData(banded).matrix(
    sort_by="band",
    sort_categories=["young", "middle", "senior"],
    title="Gaps by age band",
).show()
# mf.matrix(banded, sort_by="band", sort_categories=["young", "middle", "senior"],
#           title="Gaps by age band").show()
