# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] (2025-05-17)

Initial public release.

### Added

**Core**
* `MissingData`: central analytics class with cached missingness metrics.
* Column metrics: `col_missing_rate`, `col_missing_count`, `col_missing_percent`, `col_completeness`, `col_present_count`, `cols_complete`.
* Row metrics: `row_missing_rate`, `row_missing_count`, `row_missing_percent`, `row_completeness`, `rows_complete`, `rows_with_missing`, `rows_above_missing_threshold()`.
* Dataset metrics: `total_missing_rate`, `total_missing_count`.
* Pattern analysis: `missing_pattern_in_rows`, `missing_pattern_in_rows_unique`, `missing_pattern_counts()`, `perfectly_correlated_missing_columns()`.
* Correlation matrices: `missing_corr`, `present_present_corr`, `present_missing_corr`, `value_missing_corr`.
* Little's MCAR test: `littles_mcar_test()`.

**Plots (all return interactive Plotly figures)**
* `heatmap()`: binary row-by-column missingness matrix.
* `barchart_count()`: missing or present counts per column.
* `barchart_rate()`: missing rate per column as a bar chart.
* `barchart_total()`: total present vs missing cells across the dataset.
* `heatmap_rate()`: single-row heatmap of missing rates per column.
* `heatmap_correlation()`: phi correlation between column missingness patterns.
* `heatmap_predictive()`: correlation between presence in one column and missingness in another.
* `heatmap_biserial()`: point-biserial correlation between values and missingness indicators.
* `scatterplot(x, y)`: scatter plot that keeps missing values visible via axis offsets.
* `parallel_coordinates()`: parallel coordinates colored by missingness.
* `boxplot(x, color_by)`: distribution comparison (box or violin) split by missingness.
* `density(x, color_by)`: overlapping KDE curves split by missingness.
* `dendrogram()`: hierarchical clustering of missingness correlation.
* `barchart_venn()`: 7-region exclusive Venn subsets for 3 columns.
* `barchart_intersection()`: UpSet-style missingness intersections across columns.

**Panel**
* `Panel`: combines multiple plots in a responsive grid layout.
