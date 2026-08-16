# missingfcup

Interactive Plotly charts and cached metrics for exploring missing data in pandas
DataFrames. Give it a DataFrame; it shows you where the gaps are, whether they cluster,
and whether they relate to the values you can still see.

<!-- IMAGE — same hero shot as the README. Save to docs/images/matrix.png and replace
     this comment with: ![Missingness matrix](images/matrix.png) -->

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

There are two ways to call it. Flat functions like `mf.matrix(df)` build everything for
you and are the quickest way to look at a dataset. A `MissingData` object caches the
masks and metrics, exposes the statistical tests, and composes into a `Panel`; use it
when you work on the same DataFrame more than once.

```python
md = mf.MissingData(df)
md.matrix().show()
md.heatmap().show()
md.littles_mcar_test()
```

Every plot returns an object with two methods: `.show()` renders it, and `.save(path)`
writes it to a file, where the extension picks the format (`.html` or `.png`).

## Where to go next

* **[Plots](plots.md)** — what each visualization shows and when to reach for it.
* **[API reference](api.md)** — every metric, mask, correlation matrix and test.

## About missing data

Missing values are not just empty cells. They can bias results and hurt model
performance, and *why* a value is missing matters. The usual three mechanisms:

* **MCAR** (Missing Completely At Random): the gaps are unrelated to any data.
* **MAR** (Missing At Random): the gaps depend on other observed variables.
* **MNAR** (Missing Not At Random): the gaps depend on the missing values themselves.

You cannot prove MNAR from the data alone, but you can rule things out. `heatmap`,
`density`, `boxplot` and a sorted `matrix` are the tools for that, and
`littles_mcar_test()` puts a p-value on the MCAR question.
