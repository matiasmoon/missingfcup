"""Predictive heatmap: does observing one column predict a gap in another?

Rows are "column is present", columns are "column is missing". A strong cell means
seeing a value in one column tells you another column is likely empty, which is a
signal for MAR.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.heatmap(kind="predictive", title="Does seeing one column predict a gap in another").show()
# mf.heatmap(df, kind="predictive", title="Does seeing one column predict a gap in another").show()
