"""Scatter plot that keeps missing values on screen.

A normal scatter silently drops any row where either axis is missing. Here those
rows are offset below the axis range instead, with a distinct marker, so the
values you cannot plot stay countable.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.scatterplot(x="age", y="income", title="Age against income, gaps kept on screen").show()
# mf.scatterplot(df, x="age", y="income", title="Age against income, gaps kept on screen").show()

# Every option at once. missing_column colours the points by a third column's
# missingness, jitter separates rows that share a coordinate (the offset band needs
# it most, since every missing row sits on exactly the same spot), jitter_seed keeps
# the result reproducible, and the explicit ranges override the padded default.
md.scatterplot(
    x="age",
    y="income",
    missing_column="score",
    axis_padding=0.2,
    jitter=0.03,
    jitter_seed=7,
    xaxis_range=[10, 70],
    yaxis_range=[0, 120000],
    title="Age against income, coloured by gaps in score",
).show()
# mf.scatterplot(df, x="age", y="income", missing_column="score", axis_padding=0.2,
#                jitter=0.03, jitter_seed=7, xaxis_range=[10, 70],
#                yaxis_range=[0, 120000],
#                title="Age against income, coloured by gaps in score").show()
