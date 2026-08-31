# Plots

Every plot is a method on `MissingData` and also a flat function taking a DataFrame.
The signatures and defaults are in the [API reference](api.md); this page is about
*when* to reach for each one.

Each returns an object with `.show()` to render and `.save(path)` to write a file,
where the extension picks the format (`.html` or `.png`).

## Where are the gaps?

| Call | What it shows |
|---|---|
| `matrix()` | Row-by-column map of missing vs present. The first thing to look at. |
| `bar()` | Missing count per column. `measure="fraction"` for the fraction instead. |
| `rate()` | Missing rate as one coloured strip. Stays readable on wide datasets. |
| `totals()` | Present against missing cells for the whole dataset. |

Start with `matrix()`. If the gaps form horizontal bands, some rows are badly affected;
vertical bands mean whole columns are unreliable. A scattered dusting with no structure
is the visual signature of MCAR.

## What goes missing together?

| Call | What it shows |
|---|---|
| `heatmap()` | Correlation between columns' missingness patterns. |
| `dendrogram()` | Clusters columns by that same correlation. |
| `venn()` | The 7 exclusive missingness regions of three columns. |
| `upset()` | Every missingness intersection. Scales past the three Venn allows. |

`heatmap()` gives you the pairwise view; `dendrogram()` groups those pairs into nested
clusters, which is easier to read once you have more than a handful of columns. Use
`venn()` for exactly three columns and `upset()` for more.

## Why are they missing?

This is the part that distinguishes MCAR from MAR and MNAR.

| Call | What it shows |
|---|---|
| `heatmap(kind="predictive")` | Does *observing* one column predict a gap in another? |
| `heatmap(kind="biserial")` | Do a column's *values* relate to another column's gaps? |
| `density(column, missing_column)` | Distribution of `x`, split by whether `missing_column` is missing. |
| `boxplot(column, missing_column)` | The same split, as boxes or violins. |
| `scatterplot(x, y)` | Scatter that offsets missing values instead of dropping them. |
| `parallel_coordinates()` | All columns at once, coloured by one column's missingness. |

`heatmap(kind="biserial")` is the one to reach for first: it reads actual values rather
than just the missingness pattern, so a strong cell is direct evidence that missingness
depends on observed data, which is MAR.

Confirm what it suggests with `density()` or `boxplot()`. Two curves sitting on top of
each other mean the distribution does not change with missingness, which is consistent
with MCAR. Curves that pull apart mean it does.

For a statistical version of the same comparison, see `mann_whitney_test()` in the
[API reference](api.md).

## Shared options

Most column-based plots take the same arguments:

```python
md.bar(
    selected_columns=["age", "income"],   # draw only these
    high_missingness_threshold=0.9,       # drop columns at or above this rate
    max_columns=20,                       # cap how many appear (0 = no cap)
    sort_by="missingness",                # or "alphabetical", or None for frame order
    ascending=False,                      # as in pandas: False puts the emptiest first
).show()
```

`upset()` draws every column you name in `selected_columns`. The number of intersection
bars is capped at 20, which is not a parameter: past that the bars stop being readable.
When the cap bites, the plot draws the 20 largest intersections and warns that it did.

## Combining plots

```python
from missingfcup import Panel

Panel(
    [md.matrix(title="Matrix"), md.bar(title="Counts"), md.rate(title="Rates")],
    title="Missing data overview",
).show()
```

`Panel` arranges plots in a grid of up to `max_cols` columns. Give each plot its own
title; the panel title sits above the whole grid.
