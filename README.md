# missingfcup

`missingfcup` is a Python toolkit to inspect, quantify, and visualize missing data with Plotly.

It provides:
* A single analytics object (`MissingData`) that computes and caches missingness metrics.
* A rich set of interactive visualizations tailored to different missingness questions.
* Simple, consistent methods on `MissingData` that return interactive Plotly figures.

This makes it easy to move from diagnosis (what is missing?) to communication (how do I show it?).

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

md.heatmap().show()
```

## Example Dataset

```python
import pandas as pd
from missingfcup import MissingData

url = "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
df = pd.read_csv(url)
md = MissingData(df)

md.heatmap().show()
md.barchart_count().show()
md.heatmap_correlation().show()
```

## Visualizations

All visualizations are exposed as methods on `MissingData` and return a plot object with two methods:
* `.show()`: renders the interactive Plotly figure.
* `.save(path)`: saves the figure as an HTML file.

### Heatmap

Row-by-column binary matrix of missingness (present vs. missing).

```python
md.heatmap().show()
```

### Bar Chart: Missing Count

Counts missing (or present) values per column.

```python
md.barchart_count().show()
md.barchart_count(value="present", show_both=True).show()
```

### Bar Chart: Missing Rate

Missing rate per column as a fraction or percentage.

```python
md.barchart_rate().show()
md.barchart_rate(scale="percentage").show()
```

### Bar Chart: Dataset Total

Total present vs. missing cells across the entire dataset (two bars).

```python
md.barchart_total().show()
```

### Heatmap: Missing Rate

Single-row heatmap of missing rates per column.

```python
md.heatmap_rate().show()
md.heatmap_rate(scale="percentage").show()
```

### Heatmap: Missingness Correlation

Phi (Pearson) correlation between column missingness patterns. Columns that tend to
be missing at the same time will cluster together.

```python
md.heatmap_correlation().show()
```

### Heatmap: Predictive Correlation

Correlation between the *presence* of one column and the *missingness* of another.
Useful for diagnosing MAR: does observing a value in column A predict a gap in column B?

```python
md.heatmap_predictive().show()
```

### Heatmap: Biserial Correlation

Point-biserial correlation between column *values* and missingness indicators.
Reveals whether higher or lower values in one column associate with gaps in another.

```python
md.heatmap_biserial().show()
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

```python
md.parallel_coordinates(
    selected_columns=["A", "B", "C", "D"],
    missingness_color_column="A",
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

### Bar Chart: Venn (3 columns)

Counts the 7 exclusive missingness subsets for a 3-column Venn diagram.

```python
md.barchart_venn(selected_columns=["A", "B", "C"]).show()
```

### Bar Chart: Intersections (UpSet-style)

UpSet-style plot showing the size of every missingness intersection across columns.

```python
md.barchart_intersection(selected_columns=["A", "B", "C", "D"]).show()
```

## Panel: Combining Multiple Plots

```python
from missingfcup import Panel

panel = Panel(
    [
        md.heatmap(title="Missingness Matrix"),
        md.barchart_count(title="Missing Counts"),
        md.heatmap_rate(title="Missing Rates"),
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

Missingness is not just "empty cells"; it shapes analysis, bias, and model performance.
Three mechanisms are commonly distinguished:

* **MCAR** (Missing Completely At Random): gaps are unrelated to any data.
* **MAR** (Missing At Random): gaps depend on other observed variables.
* **MNAR** (Missing Not At Random): gaps depend on the missing values themselves.

`missingfcup` exposes all relevant views through plots and cached metrics so you can
quickly understand the structure of missing data before deciding how to handle it.

## License

MIT. See [LICENSE](LICENSE).
