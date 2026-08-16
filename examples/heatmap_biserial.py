"""Biserial heatmap: do a column's *values* relate to another column's gaps?

Point-biserial correlation between the observed values of one column and the
missingness indicator of another. Unlike the other two heatmaps, this reads the
actual numbers, not just the missingness pattern, so it needs numeric columns.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.heatmap(kind="biserial", title="Values vs missingness").show()
