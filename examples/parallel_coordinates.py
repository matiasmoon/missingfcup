"""Parallel coordinates coloured by one column's missingness.

Every row becomes a line crossing all the axes. Lines are coloured by whether the
chosen column is missing in that row, so a multivariate pattern behind the gaps
shows up as the two colours separating.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.parallel_coordinates(
    missingness_color_column="age",
    title="All columns, coloured by age missingness",
).show()
