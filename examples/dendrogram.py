"""Hierarchical clustering of columns by missingness correlation.

Columns joined low in the tree go missing together. This groups the heatmap's
pairwise view into nested clusters, which is easier to read on wide datasets.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.dendrogram(title="Columns clustered by shared gaps").show()
# mf.dendrogram(df, title="Columns clustered by shared gaps").show()

# Every option at once. linkage picks how the distance between two clusters is
# measured, and use_abs_correlation makes a strongly negative relationship count as
# close rather than far apart -- two columns that are never missing together are
# just as related as two that always are.
md.dendrogram(
    selected_columns=["age", "income", "score", "rating"],
    high_missingness_threshold=0.9,
    max_columns=4,
    drop_constant_columns=True,
    linkage="complete",
    use_abs_correlation=True,
    title="Columns clustered by shared gaps, either direction",
).show()
# mf.dendrogram(df, selected_columns=["age", "income", "score", "rating"],
#               high_missingness_threshold=0.9, max_columns=4,
#               drop_constant_columns=True, linkage="complete",
#               use_abs_correlation=True,
#               title="Columns clustered by shared gaps, either direction").show()
