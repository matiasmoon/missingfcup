"""Scatter plot that keeps missing values on screen.

A normal scatter silently drops any row where either axis is missing. Here those
rows are offset below the axis range instead, with a distinct marker, so the
values you cannot plot stay countable.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.scatterplot(x="age", y="income", title="age vs income").show()
