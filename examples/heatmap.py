"""Missingness correlation between columns.

Each cell is the correlation between two columns' missingness patterns. A high
value means the two columns tend to be empty in the same rows. This is the
equivalent of missingno's heatmap.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.heatmap(title="Which columns go missing together").show()
# mf.heatmap(df, title="Which columns go missing together").show()

# Every option the symmetric kinds take. The matrix mirrors itself, so masking the
# lower triangle drops only duplicates; drop_constant_columns removes the columns
# whose missingness never varies, whose correlation is undefined and would draw as
# a stripe of grey.
md.heatmap(
    kind="correlation",
    selected_columns=["age", "income", "score", "rating"],
    high_missingness_threshold=0.9,
    show_values=True,
    max_columns=4,
    drop_constant_columns=True,
    sort_by="missingness",
    ascending=False,
    show_upper_triangle=True,
    show_legend=False,
    title="Which columns go missing together, once each",
).show()
# mf.heatmap(df, kind="correlation", selected_columns=["age", "income", "score", "rating"],
#            high_missingness_threshold=0.9, show_values=True, max_columns=4,
#            drop_constant_columns=True, sort_by="missingness", ascending=False,
#            show_upper_triangle=True, show_legend=False,
#            title="Which columns go missing together, once each").show()
