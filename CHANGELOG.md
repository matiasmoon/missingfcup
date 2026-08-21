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

### Changed

* Parameter names now follow one written convention, recorded on
  `_MissingDataPlotMixin`. `show_*` changes what is drawn, `drop_*` changes which data
  is included, `use_*` changes how something is computed, and only one polarity of
  each switch exists. Four names changed to obey it:

  | Before | After | Why |
  |---|---|---|
  | `boxplot(plot_type=)` | `boxplot(kind=)` | `kind` is the package's word for "which variant"; `plot_type` only restated the method name |
  | `parallel_coordinates(missingness_only=)` | `parallel_coordinates(kind="values"\|"missingness")` | it selects a variant, so it belongs on `kind` rather than being the one boolean with no prefix |
  | `matrix(order_categorical=)` | `matrix(sort_categories=)` | joins `sort_by` and `ascending`, so typing `sort_` in an editor offers the whole mechanism |
  | `dendrogram(linkage_method=)` | `dendrogram(linkage=)` | dropped the package's only `_method` suffix; scipy's own argument is just `method` |

  `ascending` stays bare and unprefixed on purpose: it is taken verbatim from
  `DataFrame.sort_values`, and matching pandas is worth more than matching the table.

* Legends now appear only when they distinguish something. `bar()`, `bar(measure="rate")`
  and `venn()` draw a single series, so their one-entry legend is off; `bar(show_both=True)`
  still shows both.
* Colour bars are compact and marked only where it matters: the three heatmaps show
  -1, 0 and 1 rather than every half step, `rate()` shows just its two ends (with a
  `%` when `scale="percentage"`, matching its cells), and all of them share one size.
* Colour bars carry no title at all. The plot title says what the figure is and the
  numbers on the bar say what the colours mean.
* `matrix()` draws its colour bar as two solid blocks rather than a gradient, since
  present/missing has no in-between, and it is shown by default -- it is the only key
  that plot has.
* The density plot no longer prints y-axis numbers. A KDE integrates to 1 over the x
  range, so on a wide axis the values land near 1e-5 and rendered as `60u`; what the
  plot is read for is the shape and where the two curves separate.
* Ordering is one idea with one spelling: `sort_by` names what to sort on and
  `ascending` gives the direction, as in `pandas.sort_values`. This replaces `order`
  (a direction) and `order_by` (a list of dicts with six keys), and the 140-line
  validator behind the latter is gone.

      bar(order_by=[{"column": "__missing__", "direction": "desc"}])   ->  bar(sort_by="missingness")
      bar(order_by=[{"column": "__column__"}])                         ->  bar(sort_by="alphabetical", ascending=True)
      matrix(order_by=[{"axis": "rows", "column": "age", "ascending": True}])
                                                                       ->  matrix(sort_by="age", ascending=True)
      rate(order="desc")                                               ->  rate(sort_by="missingness")

  `sort_by` takes `"missingness"`, `"alphabetical"`, or `None` for the DataFrame's own
  order; on `matrix()` it also takes a column name, which sorts the rows by that
  column's values. `venn()` and `upset()` order intersections rather than columns, so
  theirs takes `"size"`. The `__missing__` / `__column__` / `__row__` sentinels are
  gone, and `rate()`, `heatmap()` and `matrix()` gained alphabetical ordering, which
  only `bar()` and `matrix()` could do before.
* `matrix()`'s `order_by_y_labels` is now `sort_labels`. Its `order_by_border_color`
  and `order_by_border_width` are gone: the border marks which column `sort_by` named,
  so it has to read as an annotation rather than as data, and nothing is gained by
  letting it be recoloured into the plot's own palette.
* `drop_constant_columns` now defaults to `False` on all three `heatmap()` kinds and
  on `dendrogram()`, where it previously varied by plot (`False` for correlation and
  predictive, `True` for biserial, and unset meant "depends"). A column whose
  missingness never varies includes every column with no missing values, so dropping
  those by default would hide how much of the data is intact. They are still drawn as
  the grey NaN underlay, since their association is genuinely undefined. Pass
  `drop_constant_columns=True` to leave them out.
* `rate()`'s `max_labels_with_values` is gone; the cap is fixed at 20 columns. Whether
  a number fits inside a one-row strip is a legibility limit, not a preference, and as
  an option it silently overrode an explicit `show_values=True`.
* `matrix()`'s `group_by_mode` is gone. It reversed `z` and the colour scale together,
  which cancelled out: the rendered figure was identical either way, missing stayed
  red, and the only surviving effect was the order of the two labels on the colour
  bar. Its one real consequence was a bug -- the hover text was read off `z`, so
  `group_by_mode="missing"` labelled every present cell `NA` and every missing cell
  `!NA`. Hover labels now come from the missingness mask.
* `bar()`'s `completeness_mode`, `completeness_threshold` and
  `max_columns_by_completeness` are gone. Completeness is `1 - missingness`, so the
  three of them restated, inverted, what the shared options already say, and `bar` was
  the only plot carrying them. It now picks its columns through the same helper as
  every other column plot, which also removes the last path that drew a blank chart
  instead of raising. Migrate as:

  | Before | After |
  |---|---|
  | `completeness_mode="at_least", completeness_threshold=0.9` | `high_missingness_threshold=0.1` |
  | `completeness_mode="at_most", completeness_threshold=0.9` | `sort_by="missingness", ascending=False` |
  | `max_columns_by_completeness=n` | `sort_by="missingness", max_columns=n` |

  Two differences to watch: the old threshold compared with `>=` / `<=` while
  `high_missingness_threshold` is a strict `<`, so columns sitting exactly on the
  line change side; and `max_columns_by_completeness` emitted its survivors in the
  frame's column order, where `sort_by` emits them sorted.

### Added

* `matrix(order_categorical=[...])` draws the values of the column `sort_by` names in
  exactly the order given. A nominal categorical has no inherent first or last, so
  sorting it alphabetically is an arbitrary choice; this is how the caller makes it a
  deliberate one. Values not named follow the ones that are, and missing values come
  last. `ascending` does not apply -- the sequence already states the direction, so
  reverse the sequence to reverse the plot.
* `colorscale` is gone. `rate()` runs from bare paper to `missing_color`, and the
  heatmaps run `present_color` -> white -> `missing_color`. Every pole of all three
  heatmap kinds is "more missing" against "more present", so the package now says
  that with the same two colours everywhere instead of a named plotly scale. This
  also corrects the direction: `RdBu` painted +1 (missing together) blue and -1 red.
* `missingness_color_column` is now `missing_column`. It sat one suffix away from
  `missing_color` while meaning something else entirely -- a column name, not a colour.
* `density(x, ...)` and `boxplot(x, ...)` take `column` instead of `x`. Neither plot has
  a `y`, and `boxplot` drew `x` on the *vertical* axis, so the name described a position
  it did not occupy. `scatterplot` keeps `x` and `y`, which it genuinely has.
* Three pairs of options folded into one each, removing a state in which a parameter
  silently did nothing:
  * `measure` covers what `measure` and `scale` used to say together:
    `"count"`, `"fraction"` or `"percentage"` on `bar()` and `venn()`, the last two on
    `rate()`. `scale` was ignored whenever `measure="count"`.
  * `high_missingness_threshold` replaces itself plus `ignore_high_missingness`:
    a number filters, `None` (the default) does not. The threshold did nothing unless
    the flag was on.
  * `order` replaces itself plus `order_by_missingness` on `rate()` and `heatmap()`:
    `"desc"`/`"asc"` orders by missing rate, `None` keeps the DataFrame's order.
    `order` did nothing when the flag was off.
* `venn()` and `upset()` name the columns worth comparing in the error they raise when
  `selected_columns` is missing, so the quickest path to a first look is still one step.
* `max_columns=0` means "no cap" on `bar()` too. It already did on `matrix()`,
  `rate()`, `heatmap()`, `dendrogram()` and `parallel_coordinates()` -- and it is the
  default on two of them -- but on `bar()` it produced an empty chart.
* `value_round` is gone. Every rate the package prints uses two decimals, so the same
  number reads the same way whichever plot drew it. It had been configurable on
  `rate()` and `heatmap()` (with different defaults) and hardcoded to different widths
  everywhere else.
* `max_columns` defaults to 0, meaning no cap, on every plot that takes it. `bar()` and
  `matrix()` drew at most 50 columns and `rate()` and `dendrogram()` at most 30, with
  no warning about the rest -- the same silent hiding that `ignore_high_missingness`
  used to do. Pass `max_columns=N` to cap.
* `dendrogram(line_width=3.0)` and `parallel_coordinates(line_width=2.0)` are both
  thicker, and both floats.
* `dendrogram(line_width=...)` is a float, matching `parallel_coordinates`. Same
  concept, and it was the only one typed as an int.
* `venn(value=...)` is now `venn(measure=..., scale=...)`, the same vocabulary `bar()`
  and `rate()` use. `value` meant two unrelated things across the package: on `bar` it
  picks which series to draw (`"missing"` / `"present"`), on `venn` it picked a unit
  (`"count"` / `"percent"`). `venn(value="percent")` becomes
  `venn(measure="rate", scale="percentage")`, and `measure="rate"` alone now gives a
  0-1 fraction, which venn could not produce before.
* One wording for "the columns you named cannot be drawn", and it says which ones.
  Four plots raised `No selected_columns found in DataFrame.`, `upset()` raised the
  same sentence with different spacing, and `parallel_coordinates()` used a third
  form. The message now names the offending columns and separates names that are
  absent from the DataFrame from names an earlier filter removed, because the two
  need different fixes.
* `ignore_high_missingness` now defaults to `False` everywhere. It defaulted to `True`,
  so a column that was 90% or more empty was dropped before anything was drawn -- a
  missing-data library hiding the columns with the most missing data. Pass
  `ignore_high_missingness=True` to get the old behaviour.
* `high_missingness_threshold` is 0.9 on every plot that takes it. `matrix()` used 0.95,
  so for the same DataFrame it and `bar()` disagreed about which columns existed.
* `venn()` warns when `selected_columns` names more than three columns, naming the ones
  it dropped and pointing at `upset()`. It draws the 7 exclusive regions of three sets,
  so a fourth column cannot be shown; it used to keep the first three silently.
* One vocabulary for missingness across every plot: `NA` and `!NA`, suffixed with the
  column when a named one drives the colour (`NA-age`, `!NA-age`). Previously the same
  binary was spelled four ways -- `Missing`/`Present`, `NA`/`!NA`, `age: missing`, and
  `Missing age` -- which showed up side by side inside a `Panel`.
* `legend_title` is gone from every plot. The legend entries now carry the column name,
  so the title had nothing left to say, and four plot classes no longer need to derive
  one. Colour is set through `missing_color` / `present_color` as before.
* `show_legend` is gone from `totals()`, `matrix()`, `rate()`, `heatmap()`,
  `dendrogram()` and `upset()`, which draw no legend at all -- they label their marks
  directly or use a colour bar.
* The heatmap colour bars lost their three-line HTML titles and sentence-length tick
  labels in favour of a short title and plain numbers. The sign of a correlation
  already says what those labels were spelling out.
* `matrix()`'s colour bar is labelled `NA` / `!NA` to match the legends elsewhere.
* Every plot factory on `MissingData` now spells out its presentation options
  (`title`, `width`, `height`, `background_color`, `text_color`, `missing_color`,
  `present_color`, `max_label_length`, and `show_legend` where a legend is drawn)
  instead of collecting them in `**kwargs`. They were always accepted; now an editor
  can offer them, a type checker can reject a misspelling, and an unknown keyword
  names the method you called rather than the private plot base class. Each plot
  declares only the options that actually affect it.
* `bar(completeness_mode=...)` takes `"at_least"` / `"at_most"` in place of
  `"most"` / `"least"`, which read as a ranking when they are really a comparison
  against `completeness_threshold`. The old values now raise `ValueError`; before,
  an unrecognised mode returned the frame unfiltered, which looked like the filter
  had run and matched nothing.
* Every plot factory documents all of its parameters, so the generated API
  reference covers them rather than showing a bare signature.

### Fixed

* `bar(selected_columns=["typo"])` drew an empty chart instead of raising. The bar
  charts kept their own copy of the column-selection logic, which skipped the check
  every other plot got from the shared helper; they now use that helper, so the
  filter, the selection and the error are the same code for all of them.
* Label truncation could exceed `max_label_length`. When two column names collided
  after being cut, the disambiguating suffix was appended on top of a label that had
  already used the full budget -- `venn(max_label_length=6)` produced labels of 7 and
  8 characters. The marker now comes out of the budget, and it is visible (`~2`, `~3`)
  rather than trailing spaces, which made two ticks read as duplicates on screen while
  being distinct categories to plotly.
* The dendrogram drew its outermost leaves flush against the plot frame, and the
  tallest join against the top edge. Both axes are padded now.
* Undefined cells in the association heatmaps are drawn grey but said nothing on
  hover, because the grey underlay used `hoverinfo="skip"`. They now report
  `not defined`. A cell is undefined when a column's missingness never varies, so
  the correlation cannot be computed.
* `totals()` drew a legend entry labelled `trace 0`: the plot sets `showlegend=False`,
  but `_apply_base_layout` ran afterwards and overwrote it with `show_legend=True`.
  The trace now opts out directly.
* `max_label_length` had no effect on `bar()`, `heatmap()`, `dendrogram()` and
  `parallel_coordinates()`, which drew column names at full length however long they
  were. The heatmap was the worst case, since it labels both axes. All four now
  truncate like `matrix()`, `rate()`, `upset()` and `venn()` already did.
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
