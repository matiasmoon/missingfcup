"""Missingness matrix: one row per observation, one column per variable.

Green cells are present, red cells are missing. This is the fastest way to see
whether the gaps cluster in particular rows or columns, or fall evenly.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.matrix(title="Missingness matrix").show()
