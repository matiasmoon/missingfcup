# API reference

Everything on this page is generated from the docstrings in the source, so it cannot
drift out of date. To change what appears here, edit the docstring.

## MissingData

The central class. It computes the missingness masks once and caches every metric
derived from them, so repeated access is free and the plots reuse the same values.

::: missingfcup.MissingData

## Panel

::: missingfcup.Panel

## Sample data

::: missingfcup.sample_data

## Flat functions

Each one builds a `MissingData` for you and returns the plot, which is the quickest
way to look at a DataFrame you only need to inspect once.

::: missingfcup.matrix
::: missingfcup.bar
::: missingfcup.rate
::: missingfcup.totals
::: missingfcup.heatmap
::: missingfcup.dendrogram
::: missingfcup.venn
::: missingfcup.upset
::: missingfcup.scatterplot
::: missingfcup.density
::: missingfcup.boxplot
::: missingfcup.parallel_coordinates
