# missingfcup

`missingfcup` is a Python toolkit to inspect, quantify, and visualize missing data with Plotly.

It provides:
- A single analytics object (`MissingData`) that computes and caches missingness metrics.
- A set of visualization classes tailored to different missingness questions.
- Simple, consistent methods on `MissingData` that return interactive Plotly figures.

This makes it easy to move from diagnosis (what is missing?) to communication (how do I show it?).

## Install

From the repo root:

```bash
pip install -e .
```

Or as a regular install:

```bash
pip install .
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

You can try a public dataset used in the demos:

```python
import pandas as pd
from missingfcup import MissingData

url = "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
df = pd.read_csv(url)
md = MissingData(df)

md.heatmap().show()
```

## Visualizations

All visualizations are exposed as methods on `MissingData` and return a Plot wrapper with:
- `.show()` to render the interactive figure.
- `.save(path)` to save the figure as HTML.

Below is a complete list of the built-in plots and what each one communicates.

### Missing Count Bar Chart

Counts missing (and/or present) values per column.

```python
md.missing_count_barchart().show()
```

### Missingness Heatmap

Row-by-column matrix of missingness (present vs missing).

```python
md.heatmap().show()
```

### Scatter Plot (Two Numeric Columns)

Shows how missingness aligns between two numeric variables, with missing points offset.

```python
md.scatterplot(x="age", y="income").show()
```

### Pattern Bar Chart (Row-Level Patterns)

Counts the 7 missingness patterns for a 3-column subset.

```python
md.pattern_barchart().show()
```

### UpSet Plot (Missingness Intersections)

Shows missingness intersections across multiple columns (set sizes + intersections).

```python
md.upset_plot().show()
```

### Missingness Correlation Heatmap

Correlation between columns' missingness patterns.

```python
md.missingness_correlation_heatmap().show()
```

### Column Missing Rate Heatmap

Single-row heatmap of missing rates per column (fraction or percent).

```python
md.column_missing_rate_heatmap().show()
```

### Parallel Coordinates (Missingness & Numeric Trends)

Parallel coordinates view of numeric columns, with missing values imputed and optionally colored by missingness.

```python
md.parallel_coordinates(selected_columns=["A", "B", "C"]).show()
```

## Panel (Combine Multiple Plots)

```python
from missingfcup import Panel

panel = Panel([
    md.heatmap(title="Missingness"),
    md.missing_count_barchart(title="Counts"),
    md.column_missing_rate_heatmap(title="Rates"),
])

panel.show()
```

## About Missing Data (Concepts)

Missingness is not just "empty cells" — it shapes analysis, bias, and model performance.
Common perspectives:
- **Column missingness**: which variables are sparse?
- **Row missingness**: which records are incomplete?
- **Patterns**: are the same columns missing together?
- **Structure**: are there correlated or systematic gaps?

`missingfcup` exposes all of these views through plots and cached metrics so you can
quickly understand the structure of missing data before deciding how to handle it.

## `MissingData` Analytics & Caching

`MissingData` computes core missingness masks once and caches derived metrics.
Plots and analyses reuse these cached properties to avoid redundant work and keep
results consistent across visualizations.

The API includes:
- Core masks and dimensions: `missing_mask`, `observed_mask`, `columns`, `number_of_rows`, `number_of_columns`
- Column metrics: `column_missing_rate`, `column_missing_count`, `column_missing_percent`, `column_completeness`, `column_present_count`, `columns_complete`
- Row metrics: `row_missing_rate`, `row_missing_count`, `row_missing_percent`, `row_completeness`, `rows_complete`, `rows_with_missing`
- Dataset metrics: `total_missing_rate`, `total_missing_count`
- Patterns: `missing_pattern_in_rows`, `missing_pattern_in_rows_unique`, `missing_pattern_counts()`
- Correlation: `missing_correlation()`, `perfectly_correlated_missing_columns()`
- Filters: `rows_above_missing_threshold()`, `columns_above_missing_threshold()`

### Tiny Analytics Examples

```python
md.column_missing_rate.sort_values(ascending=False).head()
md.rows_above_missing_threshold(0.2)
md.missing_pattern_counts(max_patterns=5)
```
