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

**Sample data**
* `sample_data()`: a 20x5 numeric DataFrame with structured gaps, used by the examples
  and useful for a quick experiment without loading a file.

**Project**
* `examples/` holds one runnable script per plot (previously notebooks). Each renders
  with `.show()` and writes nothing; the test suite executes all of them.
* `Makefile` with `clean`, `lint`, `fmt`, `test`, `examples` and `build` targets.
* CI additionally builds the sdist and wheel and installs the result, and runs the
  same `make` targets a developer runs locally so the two cannot drift apart.
* `kaleido>=1.0` is now required for PNG export. Older kaleido was deprecated by
  plotly and is being removed. 1.x no longer vendors Chromium, so installing the
  package is much smaller; the browser is fetched on first export, or up front with
  `plotly_get_chrome -y`.
* `plotly>=6.1.1` (was 5.20). The old floor did not actually work: kaleido 1.x
  refuses plotly below 6.1.1, and the UpSet plot uses axis properties added after
  5.20. A `minimums` CI job now installs every declared floor and runs the suite, so
  the floors are tested rather than assumed.
* scipy is a real requirement now. It was declared required but guarded as optional
  in the dendrogram and density plots; that unreachable fallback code is gone.
* The `examples` extra is renamed `notebooks`, which is what it is actually for —
  the example scripts need only the core package. It now includes jupyter (the
  notebooks import IPython and could not run without it) and drops statsmodels,
  which nothing used.
* Documentation site built with MkDocs. The API reference is generated from the
  docstrings by mkdocstrings, so it cannot drift from the code. Build it with
  `make docs`; it publishes to GitHub Pages from `main`.

### Fixed

* `order_by` now has one spec format shared by `matrix()` and `bar()`. Previously each
  read different keys, so a spec written for one was either rejected by the other or
  silently ignored: `{"direction": "desc"}` sorted ascending, and `bar()` raised
  `KeyError` on a spec `matrix()` accepted. `direction` is now honoured as an alias
  for `ascending`, and an unrecognised or contradictory key raises instead of being
  dropped.
* `upset()` warns when `max_sets` drops columns that were named in
  `selected_columns`, rather than silently drawing fewer sets than asked for.
* `upset(highlight_columns=...)` no longer fails when `highlight_color` is left unset.
* `density()` falls back to a histogram when a group has no spread, instead of raising
  `LinAlgError` out of `gaussian_kde`.

* `bar()`, `bar(measure="rate")`, `rate()`, `venn()`, `dendrogram()` and
  `parallel_coordinates()` labelled the column axis with the name of whichever column
  happened to come first, which read as though the axis were that single column. They
  now use a generic label (`"Column"`, or `"Missing columns"` for `venn()`), and the
  label follows the bars when `orientation="horizontal"`.

### Notes
* `MissingData` needs unique column names and a non-empty DataFrame.
* Most column-based plots share the same options: `selected_columns`,
  `ignore_high_missingness`, `max_columns`, and ordering by missing rate.
* The package ships type information (PEP 561 `py.typed`).
