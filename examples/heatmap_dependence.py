"""Dependence heatmap: does a column's value relate to another column's gaps at all?

The same axes as the direction heatmap, measured without a sign. Each cell is a
distance from independence on a 0-1 scale: Kolmogorov-Smirnov for numeric columns,
Cramer's V for categorical ones. Giving up the direction is what lets it see
relationships that have none, such as missingness concentrated at both tails of a
column at once, which the signed statistic reports as zero.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.heatmap(kind="dependence", title="Do values explain the gaps, either way").show()
# mf.heatmap(df, kind="dependence", title="Do values explain the gaps, either way").show()

# The two axes are set separately, as on the direction kind: values on one side,
# missingness on the other.
md.heatmap(
    kind="dependence",
    selected_value_columns=["age", "income"],
    selected_missing_columns=["score", "rating"],
    title="Do age and income explain the gaps in score and rating",
).show()
# mf.heatmap(df, kind="dependence", selected_value_columns=["age", "income"],
#            selected_missing_columns=["score", "rating"],
#            title="Do age and income explain the gaps in score and rating").show()
