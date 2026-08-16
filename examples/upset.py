"""UpSet plot of every missingness combination.

The Venn approach stops working past three columns; UpSet scales. Bars give the
size of each intersection, and the dot matrix underneath says which columns that
intersection covers.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.upset(
    selected_columns=["age", "income", "score", "rating"],
    max_sets=4,
    title="Missingness intersections",
).show()
