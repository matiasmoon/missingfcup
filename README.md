# missingfcup

`missingfcup` is a Python toolkit to inspect and visualize missing data with Plotly.

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

## Plot Functions (Very Easy Examples)

All plot functions are methods on `MissingData` and return a Plotly figure wrapper with `.show()` and `.save()`.

### Missing Count Bar Chart

```python
md.missing_count_barchart().show()
```

### Missingness Heatmap

```python
md.heatmap().show()
```

### Scatter Plot (Missingness by Two Numeric Columns)

```python
md.scatterplot(x="age", y="income").show()
```

### Pattern Bar Chart (Row-Level Missingness Patterns)

```python
md.pattern_barchart().show()
```

### Missingness Correlation Heatmap

```python
md.missingness_correlation_heatmap().show()
```

### Column Missing Rate Heatmap

```python
md.column_missing_rate_heatmap().show()
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

## Helpful Analytics on `MissingData`

These attributes and methods make it easy to inspect missingness without plotting:

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