"""Hierarchical clustering of columns by missingness correlation.

Columns joined low in the tree go missing together. This groups the heatmap's
pairwise view into nested clusters, which is easier to read on wide datasets.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.dendrogram(title="Missingness clustering").show()
