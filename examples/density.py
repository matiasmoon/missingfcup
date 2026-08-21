"""Distribution of one column split by another column's missingness.

Two overlapping KDE curves: rows where missing_column is present, and rows where it is
missing. Curves that sit on top of each other suggest MCAR; curves that pull apart
suggest MAR or MNAR.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.density(
    column="income", missing_column="age", title="Income distribution, split by gaps in age"
).show()
# mf.density(df, column="income", missing_column="age", title="Income distribution, split by gaps in age").show()
