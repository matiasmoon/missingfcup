# missingfcup

[![CI](https://github.com/matiasmoon/missingfcup/actions/workflows/ci.yml/badge.svg)](https://github.com/matiasmoon/missingfcup/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/matiasmoon/missingfcup)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Interactive Plotly charts and cached metrics for exploring missing data in pandas
DataFrames. Give it a DataFrame; it shows you where the gaps are, whether they cluster,
and whether they relate to the values you can still see.

<!-- IMAGE 1 — hero. Suggested: output of `mf.matrix(mf.sample_data())` or a real
     dataset. Save to docs/images/matrix.png and replace this comment with:
     ![Missingness matrix](docs/images/matrix.png) -->

## Install

```bash
pip install missingfcup
```

## Quick start

```python
import missingfcup as mf

df = mf.sample_data()        # or your own DataFrame
mf.matrix(df).show()
```

There are two ways to call it. Flat functions (`mf.matrix(df)`) build everything for you
and are the quickest way to look at a dataset. A `MissingData` object caches the masks
and metrics, exposes the statistical tests, and composes into a `Panel`; use it when you
work on the same DataFrame more than once.

```python
md = mf.MissingData(df)
md.matrix().show()
md.heatmap().show()
md.littles_mcar_test()
```

Every plot returns an object with two methods: `.show()` renders it, and `.save(path)`
writes it to a file, where the extension picks the format (`.html` or `.png`).

## The plots

| Call | What it shows |
|---|---|
| `matrix()` | Row-by-column map of missing vs present. The first thing to look at. |
| `bar()` | Missing count per column. `measure="fraction"` for the fraction instead. |
| `rate()` | Missing rate as one coloured strip. Stays readable on wide datasets. |
| `totals()` | Present against missing cells for the whole dataset. |
| `heatmap()` | Correlation between columns' missingness: what goes missing together. |
| `heatmap(kind="predictive")` | Does observing one column predict a gap in another? |
| `heatmap(kind="biserial")` | Do a column's *values* relate to another column's gaps? |
| `dendrogram()` | Clusters columns by missingness correlation. |
| `venn()` | The 7 exclusive missingness regions of three columns. |
| `upset()` | Every missingness intersection. Scales past the three that Venn allows. |
| `scatterplot(x, y)` | Scatter that offsets missing values instead of dropping them. |
| `density(column, missing_column)` | Distribution of `x`, split by whether `missing_column` is missing. |
| `boxplot(column, missing_column)` | The same split as boxes or violins. |
| `parallel_coordinates()` | All columns at once, coloured by one column's missingness. |

<!-- IMAGE 2 — optional gallery strip. Suggested: heatmap + upset + density side by
     side, saved to docs/images/gallery.png. Helps show the range in one glance. -->

Most column-based plots share the same options: `selected_columns` picks what to draw,
`high_missingness_threshold` drops near-empty columns, `max_columns` caps how many
appear, and `sort_by` with `ascending` orders them, as in pandas.

```python
md.bar(selected_columns=["age", "income"], measure="fraction").show()
md.upset(selected_columns=["age", "income", "score", "rating"]).show()
```

`venn()` and `upset()` need `selected_columns`: name the three columns to compare for
`venn()`, and any number of them for `upset()`.

## Combining plots

```python
from missingfcup import Panel

Panel(
    [md.matrix(title="Matrix"), md.bar(title="Counts"), md.rate(title="Rates")],
    title="Missing data overview",
).show()
```

## Metrics

`MissingData` computes the missingness masks once and caches everything derived from
them, so plots and metrics never recompute the same thing.

```python
md.col_missing_rate      # fraction missing per column
md.row_missing_count     # missing values per row
md.total_missing_rate    # overall fraction of missing cells
md.missing_corr          # correlation between column missingness patterns
md.missing_pattern_counts(max_patterns=5)
md.littles_mcar_test()
md.mann_whitney_test(x="income", by="age")
```

The full list of masks, column and row metrics, correlation matrices, pattern analysis
and statistical tests is in the **[API reference](https://matiasmoon.github.io/missingfcup/api/)**,
which is generated from the docstrings in the source.

## About missing data

Missing values are not just empty cells. They can bias results and hurt model
performance, and *why* a value is missing matters. The usual three mechanisms:

* **MCAR** (Missing Completely At Random): the gaps are unrelated to any data.
* **MAR** (Missing At Random): the gaps depend on other observed variables.
* **MNAR** (Missing Not At Random): the gaps depend on the missing values themselves.

You cannot prove MNAR from the data alone, but you can rule things out. `heatmap`,
`density`, `boxplot` and a sorted `matrix` are the tools for that, and
`littles_mcar_test()` puts a p-value on the MCAR question.

## Examples

[`examples/`](examples/) holds one short, runnable script per plot, all using the
built-in `sample_data()`. Run any of them directly:

```bash
python examples/matrix.py
```

[`notebooks/`](notebooks/) holds the fuller analyses: five datasets, each with MCAR, MAR
and MNAR missingness generated and then diagnosed with this package.

## Development

```bash
make          # list the available targets
make test     # run the test suite
make lint     # ruff check and format check
make docs     # build the documentation site
make build    # clean, then build the sdist and wheel
```

Documentation lives in [`docs/`](docs/) and is built with MkDocs. The API reference is
generated from docstrings, so changing a docstring is how you change the docs.

Saving a figure as `.png` goes through [kaleido](https://github.com/plotly/Kaleido),
which renders it in a headless browser. Kaleido downloads that browser the first time
it is needed; run `plotly_get_chrome -y` to fetch it up front. Saving to `.html` needs
none of this.

## License

MIT. See [LICENSE](LICENSE).
