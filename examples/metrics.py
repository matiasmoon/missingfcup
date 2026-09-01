"""The cached metrics, which are what every plot is drawn from.

Each one is readable on its own, so a claim a figure makes visually can be checked as a
number or asserted in a script. They are computed once from the missingness masks and
cached, so reading many of them costs no more than reading one.

This script prints rather than draws.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

# The masks everything else derives from. mask_missing is True where a value is absent,
# mask_present is its complement, and mask_observed is the numeric form.
print(f"shape: {df.shape}")
print(f"missing cells: {md.total_missing_count} of {df.size}")
print(f"overall missing rate: {md.total_missing_rate:.3f}\n")

# Per column. col_missing_rate is the one to compare across columns, since a count says
# nothing without the row total behind it.
print("per column")
print(md.col_missing_count.to_string(), "\n")
print(md.col_missing_rate.round(3).to_string(), "\n")
print(f"columns with no gaps at all: {list(md.cols_complete)}\n")

# Per row. rows_complete is worth reading early: it is what a listwise deletion would
# leave. On a wide dataset at a moderate missing rate it is routinely near zero, which
# settles whether complete-case analysis is viable before any mechanism is diagnosed.
# These return the row *labels*, not counts, so that the rows themselves can be selected
# with df.loc[...]. Wrap in len() for the count.
print(f"rows with no gaps: {len(md.rows_complete)} of {len(df)}")
print(f"rows with at least one gap: {len(md.rows_with_missing)}")
print(f"their labels: {list(md.rows_complete)}")
print(f"rows above a 50% missing rate: {list(md.rows_above_missing_threshold(0.5))}\n")

# Pattern analysis. Random deletion scatters rows into nearly as many patterns as there
# are rows; a structured mechanism concentrates them into a few, so the count is a
# mechanism signal rather than a summary.
print("the five most common missingness patterns")
print(md.missing_pattern_counts(max_patterns=5).to_string(), "\n")

# Columns whose gaps fall on exactly the same rows. One selection rule governing two
# columns produces this; independent deletion does not, except by chance. Across the five
# datasets in notebooks/, this returns pairs under MAR and nothing under MCAR or MNAR.
print(f"perfectly correlated pairs: {md.perfectly_correlated_missing_columns()}\n")

# The association matrices, one row per column. See examples/heatmap*.py for the plots
# that draw them.
print("missingness correlation, age against income")
print(f"  {md.missing_corr.loc['age', 'income']:.4f}")
print("value against missingness, visits explaining income's gaps")
print(f"  signed   {md.value_missing_corr.loc['visits', 'income']:.4f}")
print(f"  unsigned {md.value_missing_dependence.loc['visits', 'income']:.4f}")
