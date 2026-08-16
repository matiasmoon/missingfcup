"""Missingness correlation between columns.

Each cell is the correlation between two columns' missingness patterns. A high
value means the two columns tend to be empty in the same rows. This is the
equivalent of missingno's heatmap.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.heatmap(title="Missingness correlation").show()
