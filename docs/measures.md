# Choosing a measure

Every diagnostic in this package reduces to one of two questions. Either *do these two
columns go missing together?*, which reads the missingness mask alone, or *do this
column's observed values explain another column's gaps?*, which reads the data. Only the
second question can distinguish MAR from MCAR, because only it looks at anything other
than the pattern of holes.

That much is settled. What is not obvious, and what this page records, is that the second
question has more than one correct answer, and that the choice between them decides which
relationships are visible at all. This page documents how the package arrived at its
current set of measures, including two that were removed along the way. It is written for
a reader deciding which grid to trust, and for a future maintainer wondering why the
surface looks the way it does.

## 1. A measure that carried no information

The package once offered three heatmap kinds. `kind="correlation"` correlated one
column's missingness against another's. `kind="predictive"` correlated one column's
*presence* against another's missingness, and was documented as the plot that
distinguishes MCAR from MAR.

It was neither. Presence and missingness are one fact recorded twice: a cell is either
empty or it is not, and `mask_present` is defined as `~mask_missing`. Pearson correlation
is unchanged by reflecting a variable and only changes sign, so for every pair of columns

```
corr(1 - miss_i, miss_j) == -corr(miss_i, miss_j)
```

The predictive grid was therefore the correlation grid with its sign flipped, cell for
cell. Checked on the packaged sample data, the two matrices agreed to sixteen decimal
places:

```
is predictive == -correlation ?   True
max |corr + predictive|           1.1e-16
```

Two consequences followed. The first is that a reader studying both grids side by side,
as the analyses in this repository did, was reading one grid twice and counting it as two
pieces of evidence. One notebook stated this explicitly, describing the predictive
reading as corroborating the value-based one, when it could not corroborate anything: it
never read a value.

The second is that the documentation had filed it under the MAR question. A grid computed
entirely from the missingness mask cannot separate MAR from MCAR, however it is signed or
labelled, because the mask does not record what the missing values would have been or
what the observed ones are. The name promised evidence the computation could not supply.

A second metric, `present_present_corr`, was removed for the same reason. Correlating two
presence masks returns the missingness correlation unchanged, since `corr(1 - x, 1 - y)`
equals `corr(x, y)`; it was public, and no plot ever called it.

The lesson worth carrying: a measure that is an algebraic transform of another measure
adds a picture without adding a fact. We now state this property in the `missing_corr`
docstring so that the question does not have to be rediscovered.

## 2. What a signed correlation cannot see

The remaining value-reading grid computed a point-biserial correlation: a Pearson
correlation between a column's observed values and another column's binary missingness
indicator. It is the natural first choice, and its sign is genuinely useful, since it
says whether *higher* or *lower* values accompany the gaps.

The sign is also its limit. A correlation measures a shift in location, so it can only
detect relationships that have a direction. When missingness is concentrated at both ends
of a column's range, the two ends cancel: the present group and the missing group end up
with the same mean, and the correlation reports approximately zero for a relationship that
is complete.

We constructed the extreme case to measure how badly this fails. In a frame of 2000 rows,
`income` was made missing exactly when `age` fell in its outer 30 per cent, so that
missingness is a deterministic function of an observed column:

```
point-biserial                    0.0084
mann_whitney_test p-value         1.0000  (not significant)
Kolmogorov-Smirnov p-value        3e-96
```

The package's two existing tools both reported nothing. That `mann_whitney_test` also
failed is worth dwelling on, because it looks like the obvious fallback: it is
non-parametric, makes no normality assumption, and is otherwise a more robust instrument
than a Pearson correlation. It failed anyway, and for the same reason. Mann-Whitney
compares stochastic ordering, which is still a statement about where the two groups sit
relative to each other. Being more careful about the wrong question does not answer a
different one.

This configuration is not exotic. A sensor that clips at both its limits produces it. So
does a survey question skipped by respondents at either extreme of a scale, and any
filter applied to the middle of a distribution rather than to one of its tails.

## 3. Comparing distributions rather than centres

The two-sample Kolmogorov-Smirnov statistic compares the two groups' empirical
distribution functions at every point and reports their largest separation. It has no
direction to report, which is precisely what allows it to see relationships that have
none. On the case above it returns 0.5, and its p-value is 3e-96.

Two properties made it adoptable rather than merely correct.

It does not disagree with the measure it supplements. On ordinary one-sided relationships
the two statistics rank the column pairs the same way, so widening the net costs nothing
in the cases that already worked. It is a superset of what a location test finds, not a
rival answer.

It is also cheap, once implemented properly. A naive implementation calling
`scipy.stats.ks_2samp` once per column pair took 24 seconds on a 30000-row, 24-column
frame, roughly fifty times the cost of the point-biserial matrix it would sit beside.
Sorting each value column once and walking the two cumulative distributions together
across all missingness columns at once reduces this to 0.26 seconds, which is faster than
the point-biserial loop was before we vectorised that too. The implementation agrees with
scipy exactly, including the handling of tied values, which requires comparing the two
distribution functions only at the end of each run of equal values rather than inside it.

The same measure is exposed as `ks_test(x, by)` beside `mann_whitney_test(x, by)`. The
pair is deliberate: a significant `ks_test` against a flat `mann_whitney_test` is the
signature of a relationship with no direction, and having both lets a reader recognise it
rather than merely stumble over it.

## 4. Columns whose numbers are names

The second failure has nothing to do with tails. A great many missing-data problems
involve categorical variables, and a point-biserial correlation cannot read one at all: a
string column is coerced to `NaN` and the cell comes back blank.

The obvious remedy is to encode the categories as integers first, which is what this
package's own documentation recommended. For an ordinal variable that is correct. For a
nominal one it is worse than doing nothing, because the resulting number is real,
plausible, and meaningless.

We demonstrated this on the `contraceptive_method` analysis, where `husband_occupation`
holds four occupation categories numbered 1 to 4 and the numbering carries no order.
Relabelling those four codes in all 24 possible ways, and recording what each measure
reports for the same underlying data:

| Measure | Lowest | Highest | Spread |
|---|---|---|---|
| direction, float codes (point-biserial) | −0.1241 | +0.1241 | 0.2482 |
| dependence, float codes (Kolmogorov-Smirnov) | 0.0655 | 0.1325 | 0.0670 |
| dependence, `category` dtype (Cramér's V) | 0.1331 | 0.1331 | 0.0000 |

The signed correlation swings across a quarter of its range and passes close to zero,
driven by nothing but the numbering. Since `pandas.factorize` assigns codes in order of
first appearance, the value it produces is a function of how the rows happened to be
sorted.

The unsigned reading on the same float codes improves matters without fixing them. It
takes the Kolmogorov-Smirnov branch, because the dtype is numeric and the package cannot
know the numbers are names. Its spread is smaller, and it never approaches zero: a genuine
departure from independence cannot make two distribution functions coincide under any
relabelling, so it will not report *nothing* where something exists. Its magnitude,
however, is not comparable across labellings.

Cramér's V is computed from a contingency table and never consults an ordering, so it
returns the same value for all 24 labellings. Because the missingness side has exactly two
levels, it reduces to `sqrt(chi2 / n)`.

The practical rule is therefore to store nominal variables as `category` or as strings and
let the package choose, rather than to factorize them first. We have corrected the advice
in `MissingData`'s own docstring accordingly.

## 5. Why this became a third kind and not a parameter

Having settled on two statistics for the value-reading question, the obvious design was
one grid with a `statistic=` argument. We rejected it for three reasons.

Most of the combinations it would offer are not real. Cramér's V applies to categorical
columns and Kolmogorov-Smirnov to numeric ones, and the column's dtype already determines
which is correct. That is dispatch, not a preference, and asking a caller to state it
invites them to state it wrongly.

The remaining choice, between the signed and unsigned readings, is real, however it is not
only a choice of statistic. A point-biserial correlation runs from −1 to 1 around a
meaningful zero and requires a diverging colour scale with two poles. The unsigned
measures run from 0 to 1, where zero is independence and belongs at the end of the scale
rather than in its middle. A single grid cannot carry both without either mixing
incompatible scales in one legend or inventing a negative half that cannot occur.

Finally, `statistic=` would have applied only to the value-reading kinds, adding a fourth
parameter that raises on the third kind. The heatmap already carries three such
parameters out of twenty, and each one is a place where the signature promises a
combination the implementation refuses.

Treating the two as separate kinds makes each one internally coherent: one statistic, one
scale, one legend, one set of applicable options.

## 6. The result

| `kind` | Compares | Rows used | Measure | Range |
|---|---|---|---|---|
| `"correlation"` (default) | one column's missingness against another's | every row | Pearson on two booleans, which is the phi coefficient | −1 … +1 |
| `"direction"` | one column's values against another's missingness | rows where the value column is observed | point-biserial correlation | −1 … +1 |
| `"dependence"`, numeric | the value column split into two groups by the other's missingness | rows where the value column is observed | two-sample Kolmogorov-Smirnov statistic | 0 … 1 |
| `"dependence"`, categorical | the value column's categories against the other's missingness | rows where the value column is observed | Cramér's V | 0 … 1 |

The reading order matters and is the opposite of the order in which these were built. We
recommend `"dependence"` first, to establish whether a relationship exists at all, then
`"direction"` to describe which way it runs. Consulting only the signed grid and stopping
at a flat cell is how a real mechanism is missed.

`kind="biserial"`, the former name of `"direction"`, raises rather than aliasing. The old
name described the statistic, and the grid now offers two, so the name no longer said
which one a caller would receive.

## 7. What is still not solved

Three limitations are worth stating rather than leaving to be discovered.

A nominal variable stored as integer codes still takes the Kolmogorov-Smirnov branch,
because dtype is the only signal available and integers are a legitimate numeric type. The
result is usable and never falsely zero, however it is not the label-invariant reading.
Only storing the column as `category` obtains that, and the analyses in this repository
read their columns as their CSV files store them.

Cramér's V and the Kolmogorov-Smirnov statistic both occupy 0 to 1 and both measure
distance from independence, which is what makes it defensible to draw them on one scale.
They are not the same quantity, and a cell of 0.4 computed one way is not exactly
comparable to a cell of 0.4 computed the other. The grid is a screening instrument, and a
cell worth acting on is worth confirming with `ks_test`, `density` or `boxplot`.

Every measure here is marginal: it relates one column to one other. Missingness in two
columns may both be driven by a third, and none of these grids separates a direct
relationship from an induced one. A partial-association view would, and the package does
not currently offer one.
