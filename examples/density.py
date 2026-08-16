"""Distribution of one column split by another column's missingness.

Two overlapping KDE curves: rows where color_by is present, and rows where it is
missing. Curves that sit on top of each other suggest MCAR; curves that pull apart
suggest MAR or MNAR.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.density(x="income", color_by="age", title="income by age missingness").show()
