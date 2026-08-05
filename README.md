# missingfcup

`missingfcup` is a Python package for looking at missing data. You give it a pandas
DataFrame and it gives you interactive Plotly charts and cached metrics that show where the
gaps are and how they relate to each other.

You can call it two ways:

* Flat functions like `mf.matrix(df)`, `mf.heatmap(df)` and `mf.bar(df)`, for a quick look.
  If you already use missingno, these will feel familiar.
* A `MissingData` object, which caches the metrics, exposes statistical tests, and lets you
  combine several plots into a `Panel`.

```python
import missingfcup as mf

mf.matrix(df)                 # flat: builds a MissingData for you and renders inline
md = mf.MissingData(df)       # object: caching, metrics, tests, Panel
md.matrix().show()
```

## Install

```bash
pip install missingfcup
```

## Quick Start

```python
import pandas as pd
from missingfcup import MissingData

df = pd.read_csv("your_data.csv")
md = MissingData(df)

md.matrix().show()
```

## Example Dataset

```python
import pandas as pd
from missingfcup import MissingData

url = "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
df = pd.read_csv(url)
md = MissingData(df)

md.matrix().show()
md.bar().show()
md.heatmap().show()          # missingness correlation
```

## Visualizations

All visualizations are exposed as methods on `MissingData` (and as flat functions), and
return a plot object with two methods:
* `.show()`: renders the interactive Plotly figure.
* `.save(path)`: saves the figure to a file. The extension picks the format, `.html` or `.png`.

Most of the column-based plots take the same options: `selected_columns` to pick which
columns to show, `ignore_high_missingness` to drop the near-empty ones, `max_columns` to cap
how many are drawn, and ordering by missing rate.

### Matrix

Row-by-column binary matrix of missingness (present vs. missing). Equivalent to `msno.matrix`.

```python
md.matrix().show()
```

### Bar Chart

Per-column missingness as counts (`measure="count"`, the default) or rate (`measure="rate"`).
Equivalent to `msno.bar`.

```python
md.bar().show()                                    # missing counts per column
md.bar(value="present", show_both=True).show()     # present vs missing counts
md.bar(measure="rate").show()                      # missing rate (fraction)
md.bar(measure="rate", scale="percentage").show()
```

### Totals

Two bars with the total present and missing cells in the whole dataset. Useful when you
just want the overall number.

```python
md.totals().show()
```

### Rate

The missing rate per column, drawn as a single colored row. This stays compact when you
have many columns, where a bar chart would get crowded.

```python
md.rate().show()
md.rate(scale="percentage").show()
```

### Heatmap: Association

Association between column missingness patterns, selected via `kind`. Equivalent to
`msno.heatmap` (which is the `"correlation"` kind).

* `kind="correlation"` (default): Phi (Pearson) correlation between column missingness
  patterns. Columns that tend to be missing at the same time cluster together.
* `kind="predictive"`: correlation between the *presence* of one column and the
  *missingness* of another. Useful for diagnosing MAR: does observing a value in column A
  predict a gap in column B?
* `kind="biserial"`: point-biserial correlation between column *values* and missingness
  indicators. Reveals whether higher or lower values in one column associate with gaps in
  another. Accepts `selected_value_columns` / `selected_missing_columns`.

```python
md.heatmap().show()                     # correlation (default)
md.heatmap(kind="predictive").show()
md.heatmap(kind="biserial").show()
```

### Scatter Plot

Scatter plot of two numeric columns. Missing values are offset below the axis range
so they remain visible rather than being dropped.

```python
md.scatterplot(x="age", y="fare").show()
```

### Parallel Coordinates

Parallel coordinates view of numeric columns, colored by the missingness status of
a chosen column. Useful for spotting multivariate patterns associated with missing data.
Use `max_columns` to cap the number of axes when a dataset is wide.

```python
md.parallel_coordinates(
    selected_columns=["A", "B", "C", "D"],
    missingness_color_column="A",
    max_columns=8,
).show()
```

### Box and Violin Plot

Compares the distribution of column `x` between rows where `color_by` is present vs. missing.
Diverging distributions suggest MAR or MNAR; overlapping suggests MCAR.

```python
md.boxplot(x="fare", color_by="age").show()
md.boxplot(x="fare", color_by="age", plot_type="violin").show()
```

### Density Plot

Overlapping KDE curves of column `x` split by the missingness of `color_by`.

```python
md.density(x="fare", color_by="age").show()
```

### Dendrogram

Hierarchical clustering of columns by missingness correlation.
Columns that cluster together tend to be missing in the same rows.

```python
md.dendrogram().show()
```

### Venn (3 columns)

Counts the 7 exclusive missingness subsets for a 3-column Venn diagram.

```python
md.venn(selected_columns=["A", "B", "C"]).show()
```

### UpSet

UpSet plot showing the size of every missingness intersection across columns.

```python
md.upset(selected_columns=["A", "B", "C", "D"]).show()
```

## Panel: Combining Multiple Plots

```python
from missingfcup import Panel

panel = Panel(
    [
        md.matrix(title="Missingness Matrix"),
        md.bar(title="Missing Counts"),
        md.rate(title="Missing Rates"),
    ],
    title="Missing Data Overview",
)
panel.show()
```

## MissingData Analytics API

`MissingData` computes core missingness masks once and caches all derived metrics.
Plots and analyses reuse these cached properties to avoid redundant work.

### Core masks

| Property | Description |
|---|---|
| `mask_missing` | Boolean DataFrame where True means the cell is NaN |
| `mask_present` | Boolean DataFrame where True means the cell has a value |
| `mask_observed` | uint8 NumPy array where 0 = missing and 1 = present |

### Column metrics

| Property | Description |
|---|---|
| `col_missing_rate` | Fraction missing per column (0.0 to 1.0) |
| `col_missing_count` | Count of missing values per column |
| `col_missing_percent` | Percentage missing per column (0 to 100) |
| `col_completeness` | Fraction present per column (complement of col_missing_rate) |
| `col_present_count` | Count of present values per column |
| `cols_complete` | Column labels with zero missing values |

### Row metrics

| Property | Description |
|---|---|
| `row_missing_rate` | Fraction missing per row (0.0 to 1.0) |
| `row_missing_count` | Count of missing values per row |
| `row_missing_percent` | Percentage missing per row (0 to 100) |
| `row_completeness` | Fraction present per row |
| `rows_complete` | Index of rows with no missing values |
| `rows_with_missing` | Index of rows with at least one missing value |

### Dataset totals

| Property | Description |
|---|---|
| `total_missing_rate` | Overall fraction of missing cells (0.0 to 1.0) |
| `total_missing_count` | Total number of missing cells |

### Correlation matrices

| Property | Description |
|---|---|
| `missing_corr` | Pearson correlation between column missingness indicators |
| `present_present_corr` | Pearson correlation between column presence indicators |
| `present_missing_corr` | Correlation between presence in one column and missingness in another |
| `value_missing_corr` | Point-biserial correlation between column values and missingness indicators |

### Pattern analysis

```python
md.missing_pattern_in_rows            # per-row tuple of missing column names
md.missing_pattern_in_rows_unique     # unique patterns observed
md.missing_pattern_counts(max_patterns=5)
md.perfectly_correlated_missing_columns()
```

### Statistical test

```python
md.littles_mcar_test()   # Little's MCAR test (returns chi2, df, p_value, ...)
```

### Filtering helpers

```python
md.rows_above_missing_threshold(0.2)  # rows missing more than 20% of values
```

## About Missing Data

Missing values are not just empty cells. They can bias your results and hurt model
performance, and the reason a value is missing matters. People usually talk about three
mechanisms:

* **MCAR** (Missing Completely At Random): the gaps are unrelated to any data.
* **MAR** (Missing At Random): the gaps depend on other observed variables.
* **MNAR** (Missing Not At Random): the gaps depend on the missing values themselves.

`missingfcup` gives you the plots and metrics to see which of these you are dealing with,
before you decide how to handle the gaps.

## License

MIT. See [LICENSE](LICENSE).
