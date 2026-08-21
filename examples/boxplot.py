"""The same comparison as density, drawn as boxes or violins.

Box plots make the medians and quartiles easy to compare; violins also show the
shape of each distribution. Both answer: does this column's distribution shift
depending on whether another column is missing?

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.boxplot(
    column="income", missing_column="age", title="Income spread, split by gaps in age"
).show()
# mf.boxplot(df, column="income", missing_column="age", title="Income spread, split by gaps in age").show()

md.boxplot(
    column="income", missing_column="age", kind="violin", title="Income shape, split by gaps in age"
).show()
# mf.boxplot(df, column="income", missing_column="age", kind="violin", title="Income shape, split by gaps in age").show()
