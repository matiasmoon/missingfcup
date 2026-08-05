# Changelog

All notable changes to this project are written here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions can still break
things.

## [0.1.0] - Unreleased

First version. Not on PyPI yet.

### Added

**Two ways to call it**
* Flat functions for a quick look: `matrix(df)`, `heatmap(df)`, `bar(df)`, and the rest.
  Each one builds a `MissingData` for you and renders inline in a notebook.
* A `MissingData` object for repeated work on the same DataFrame: cached masks and metrics,
  statistical tests, and `Panel` composition.

**Core (`MissingData`)**
* Cached missingness masks: `mask_missing`, `mask_present`, `mask_observed`.
* Column metrics: `col_missing_rate`, `col_missing_count`, `col_missing_percent`,
  `col_completeness`, `col_present_count`, `cols_complete`.
* Row metrics: `row_missing_rate`, `row_missing_count`, `row_missing_percent`,
  `row_completeness`, `rows_complete`, `rows_with_missing`, `rows_above_missing_threshold()`.
* Dataset metrics: `total_missing_rate`, `total_missing_count`.
* Pattern analysis: `missing_pattern_in_rows`, `missing_pattern_in_rows_unique`,
  `missing_pattern_counts()`, `perfectly_correlated_missing_columns()`.
* Correlation matrices: `missing_corr`, `present_present_corr`, `present_missing_corr`,
  `value_missing_corr`.
* Statistical tests: `littles_mcar_test()`, `mann_whitney_test()`.

**Plots** (methods on `MissingData`, also available as flat functions; all return
interactive Plotly figures)
* `matrix()`: binary row-by-column missingness matrix (nullity matrix).
* `heatmap(kind=...)`: association between column missingness. `kind="correlation"`
  (default), `"predictive"`, or `"biserial"`.
* `bar(measure=...)`: per-column missingness bars. `measure="count"` (default) or `"rate"`.
* `rate()`: the missing rate per column as a single colored row.
* `totals()`: present vs. missing cells for the whole dataset.
* `upset()`: UpSet plot of missingness intersections across columns.
* `venn()`: the 7 exclusive missingness subsets for 3 columns.
* `dendrogram()`: hierarchical clustering of missingness correlation.
* `scatterplot(x, y)`: scatter that keeps missing values visible by offsetting them.
* `density(x, color_by)`: overlapping KDE curves split by missingness.
* `boxplot(x, color_by)`: box or violin distributions split by missingness.
* `parallel_coordinates()`: parallel coordinates colored by missingness, with a
  `max_columns` cap on the number of axes.

**Panel**
* `Panel`: combines several plots into one grid.

### Notes
* `MissingData` needs unique column names and a non-empty DataFrame.
* Most column-based plots share the same options: `selected_columns`,
  `ignore_high_missingness`, `max_columns`, and ordering by missing rate.
* The package ships type information (PEP 561 `py.typed`).
