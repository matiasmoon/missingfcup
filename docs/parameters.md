# Parameters

The [API reference](api.md) is generated from the docstrings and gives you every
signature and every default. This page answers the question that a signature cannot:
*why does this parameter exist at all*, why does it behave the way it does, and what is
it supposed to do for the person reading the figure.

We have written it because a plotting library accumulates options faster than it
accumulates reasons for them, and an option whose reason nobody can state is an option
that will eventually be defaulted wrong. Everything below is recovered from the code,
the docstrings, the CHANGELOG or the example scripts. Where a reason is genuinely not
recorded anywhere, we say so rather than inventing one.

## How the names are chosen

The parameter names are the package's public surface, so the convention is written down
on `_MissingDataPlotMixin` and every new option follows it. A `show_*` flag changes what
is **drawn** and never the numbers; a `drop_*` flag changes which **data** is included;
a `use_*` flag changes **how something is computed**. A `selected_*` option is a list of
column names the caller supplies, `max_*` is a cap where `0` means no cap, `*_threshold`
is a float compared against a rate, `*_color` is a colour, `*_range` is an explicit
`[min, max]`, `kind` selects a variant of the plot, and `measure` selects the quantity
shown. Anything left is a bare noun for the one obvious property, such as `title` or
`width`.

The three boolean prefixes carry most of the load. A reader can tell from the prefix
alone whether a flag is cosmetic or whether it moves the numbers, which matters more in
a missing-data library than in a general plotting library: hiding a column here is not a
style choice, it is a claim about the dataset. Only one polarity of each switch exists.
There is no `include_*` sitting opposite `drop_*`, because two spellings of one switch is
how they end up disagreeing.

`ascending` is the one deliberate exception. It is bare and unprefixed because it is
taken verbatim from `DataFrame.sort_values`, and matching pandas is worth more than
matching our own table.

One rule cuts across all of this: a parameter that cannot apply to the call being made
raises rather than being ignored. `show_both` on a rate measure, `show_upper_triangle`
on the asymmetric heatmap and `sort_categories` without a sort column all refuse the
call, because silently dropping an argument looks exactly like honouring it.

## Shared column options

Most column-based plots decide which columns to draw through one helper,
`select_columns` in `missingfcup/plots/_selection.py`. That logic used to be copied into
each plot, which is why `bar(selected_columns=["typo"])` once drew an empty chart while
every other plot raised. It now lives in one place, so the filter, the selection and the
error message are the same code for all of them.

### The order of the filters

The steps run in a fixed order, and the order is itself the design decision worth stating
once:

1. drop columns at or above `high_missingness_threshold`;
2. keep the columns named in `selected_columns`;
3. drop columns whose missingness never varies, if `drop_constant_columns` is set;
4. sort by `sort_by` in the direction `ascending` gives;
5. cap the survivors at `max_columns`.

The consequence that matters is that the threshold runs *before* the selection. A column
the threshold removed stays removed even if you name it explicitly, and if nothing you
named survives, the call raises rather than drawing a smaller figure that looks complete.
We chose this order because the threshold is a statement about the data ("these columns
are too empty to be worth drawing") while the selection is a statement about the figure,
and a data-level filter should not be silently overridden by a presentation-level one.

When `selected_columns` names columns that cannot be drawn, the error separates names
that are absent from the DataFrame from names an earlier filter removed. The two need
different fixes, and one message that says neither is worse than two that say which.

### `selected_columns`

Restricts the plot to the columns you name. It exists because the useful figure is
usually a subset: a fifty-column frame drawn in full is a texture, not a reading.

It defaults to `None`, meaning every column, on every plot that treats it as optional.
Two plots make it required instead. `venn()` needs exactly three columns because a
three-set Venn has exactly seven exclusive regions and no other number of columns
defines the plot; `upset()` requires it because choosing the columns on the caller's
behalf would mean the figure quietly answered a different question from the one asked.
Both raise with a suggestion attached — `venn()` names the three emptiest columns and
`upset()` lists every column with missing values — so being required does not make the
first look any slower.

For the reader of the plot, this is what keeps the figure about the columns under
investigation rather than about the frame's incidental width.

### `high_missingness_threshold`

Drops every column whose missing rate reaches the given value, before anything else
happens. The comparison is a strict `<` on the keep side, so a column sitting exactly on
the threshold is dropped.

It exists because a column that is 98% empty flattens the scale of every plot it appears
in, and there are datasets where that column is a known artefact rather than a finding.
This one option replaced a pair, `high_missingness_threshold` plus
`ignore_high_missingness`: the threshold did nothing unless the flag was on, which is
one of the states we removed throughout the API. A number now filters and `None` does not.

The default is `None`, meaning nothing is dropped, and that default is deliberate. It
used to be `0.9` with the flag on, so a column that was 90% or more empty was hidden
before anything was drawn — a missing-data library hiding the columns with the most
missing data. The emptiest columns are usually exactly what a missing-data plot is for.

For the reader, this is the switch that says "I already know about that column, show me
the rest", and its default guarantees that nothing was quietly removed unless you asked.

### `drop_constant_columns`

Drops columns whose missingness never varies — those that are always present, or always
missing. It exists because a correlation against a constant is undefined: such a column
contributes a row and a column of blank cells to a heatmap and nothing at all to a
clustering.

Only `heatmap()` and `dendrogram()` expose it, which are the two plots that compute a
correlation. It defaults to `False` on both. The default used to vary by plot, which
meant the same DataFrame produced different column sets depending on which figure you
asked for. We settled on `False` because "missingness never varies" includes every column
with no missing values at all, and dropping those by default would hide how much of the
data is intact. They are still drawn, as the grey undefined underlay, since their
association is genuinely not computable and the grey says so; hovering an undefined cell
reports `not defined` rather than staying silent.

For the reader, leaving it off keeps the complete columns visible as context. Turning it
on is the right move once you have seen them and want the correlated block on its own.

### `sort_by` and `ascending`

Ordering is one idea with one spelling, borrowed from pandas: `sort_by` names what to
sort on and `ascending` gives the direction. This pair replaced `order` (a direction) and
`order_by` (a list of dictionaries with six keys), along with the 140-line validator
behind the latter and the `__missing__` / `__column__` / `__row__` sentinels it needed.

`sort_by` takes `"missingness"`, `"alphabetical"`, or `None` for the DataFrame's own
column order. `matrix()` additionally accepts a column name, which switches the parameter
from ordering columns to ordering rows; `venn()` and `upset()` order regions rather than
columns, so theirs takes `"size"`. The rewrite also gained `rate()`, `heatmap()` and
`matrix()` an alphabetical order, which only `bar()` and `matrix()` could do before.

`ascending` defaults to `False` everywhere. On `"missingness"` that puts the emptiest
columns first, which is the reading order for a missing-data figure: the columns you are
most likely to act on are nearest the origin. Note that the same default applied to
`"alphabetical"` gives Z to A, because `ascending` means what it means in pandas rather
than what happens to look tidy; pass `ascending=True` for A to Z.

The default for `sort_by` itself is not uniform, and the split follows from what the
axis means on each plot rather than from inconsistency.

On `rate()` and `heatmap()` the default is `"missingness"`. Both draw columns against
columns, the order carries no information of its own, and the reading that matters is
which columns are emptiest, so the plot ranks them.

On `venn()` and `upset()` the default is `"size"`. Neither orders columns at all: they
order missingness *regions* and *intersections*, which have a natural magnitude and no
other candidate ordering. `"size"` is the only value those two accept.

On `bar()`, `matrix()` and `parallel_coordinates()` the default is `None`, which keeps
the DataFrame's own column order. In each of the three the position of a column is part
of what the reader is looking at. `matrix()` draws the frame itself, so reordering its
columns silently redraws the dataset; `parallel_coordinates()` reads as a sequence of
axes, and neighbouring axes are compared against each other; and `bar()` puts categories
on an axis where the caller's ordering is frequently deliberate. We default these to the
order that was passed in, and let the caller ask for a ranking, because the reverse
default would mean the figure quietly disagreed with the frame it came from.

The rule underneath: sort by default where column order is arbitrary, preserve it where
the order is data.

### `max_columns`

A hard cap on how many columns are drawn, applied last so that it takes the first N of
whatever the sort produced.

It defaults to `0`, meaning no cap, on every plot that takes it. This is a corrected
default rather than an original one: `bar()` and `matrix()` used to draw at most 50
columns and `rate()` and `dendrogram()` at most 30, with no warning about the rest. That
is the same silent hiding the old `ignore_high_missingness` did, and it is worse on a cap
than on a threshold, because a truncated figure looks exactly like a complete one.
`max_columns=0` also had to be taught to `bar()`, where it previously produced an empty
chart instead of an uncapped one.

The pairing with `sort_by` is where the parameter earns its place: `sort_by="missingness",
max_columns=20` is "the twenty worst columns", which is a question, whereas
`max_columns=20` alone is just the first twenty in frame order.

For the reader, the cap is what keeps a wide dataset legible. It is worth remembering
that it does not announce itself, so a capped figure should carry a title that says so.

## Shared style options

Every plot factory spells out its presentation options rather than collecting them in
`**kwargs`. They were always accepted; declaring them means an editor can offer them, a
type checker can reject a misspelling, and an unknown keyword names the method you called
instead of a private base class. Each plot declares only the options that actually affect
it, which is why `totals()`, `dendrogram()` and `upset()` have no `show_legend`.

### `title`

The title drawn above the figure, `None` by default. It has a second job worth knowing
about: it also names the PNG that the toolbar's download button writes, through
`_download_filename`, which slugifies it and prefixes the plot's own class name. A figure
saved from the browser therefore arrives with a filename that says what it is, provided
you titled it.

Inside a `Panel` the individual titles become the subplot titles and the panel's own
title sits above the grid, so giving each plot a title is what makes a panel readable.

### `width` and `height`

Figure size in pixels, defaulting to 900 by 520 and capped in `_Plot.__init__` at 2000 by
1000. The cap exists so that a mistyped size cannot produce a figure that no browser will
render usefully.

The particular numbers — 900, 520, and the two caps — are not justified anywhere in the
code or the history. They read as a reasonable notebook-width default and a generous
ceiling, and nothing more definite can honestly be said about them.

`width` has one non-cosmetic effect: when `max_label_length` is `0`, the label budget is
derived from it.

### `background_color` and `text_color`

The paper and plot background, and the font colour. Both default to `None`, which keeps
the plotly default rather than imposing a theme of our own. We default this way because a
figure is usually embedded in something — a notebook with its own theme, a slide deck, a
thesis — and a library that hardcodes white paper is a library that produces figures with
the wrong background in half of those places.

### `missing_color` and `present_color`

The two colours the whole package speaks in, defaulting to `#d62728` for missing and
`#2ca02c` for present.

This pair replaced a `colorscale` option, and the replacement is the substantive decision.
Every pole of every scale in the package is "more missing" against "more present":
`rate()` runs from bare paper to `missing_color`, the association heatmaps run
`present_color` to white to `missing_color`, and `matrix()` uses the two as flat blocks.
Saying that with two named colours rather than a named plotly scale means one change
recolours every figure consistently, and it corrected a direction error along the way —
`RdBu` painted +1, meaning "missing together", blue and -1 red, which is backwards for
this domain.

The particular hex values are `tab10[3]` and `tab10[2]`, the red and green of the
matplotlib default colour cycle, which we have confirmed against matplotlib rather than
inferred. Adopting them means a figure from this package sits beside a default matplotlib
or seaborn figure without a visible change of palette. No comment or CHANGELOG entry
recorded this at the time, so it is documented here rather than in the source history.

Two colours nearby are deliberately *not* configurable, and the code says why in both
cases. The grey `#c7c7c7` used for undefined heatmap cells must sit off the
green-white-red ramp entirely, because an undefined cell painted in `missing_color` would
read as the strongest possible positive association — the one reading it must never have.
The blue `#1f77b4` border that `matrix()` draws around the column `sort_by` names is
fixed for the same class of reason: it has to read as annotation rather than as data, so
it is deliberately not one of the plot's own colours.

### `show_legend`

Whether to draw the legend, or on the plots that use one instead, the colour bar.
It defaults to `True`.

The rule we settled on is that a legend appears only when it distinguishes something.
Several plots draw a single series, so their one-entry legend would repeat what the title
already says; those traces opt out at the trace level. The practical consequence is that
`show_legend` is inert on `bar()` unless `show_both=True`, on `bar(measure=...)`, on
`venn()`, and on `parallel_coordinates()` when no `missing_column` is given — in each of
those cases there is one series and no legend to show. It does real work on
`scatterplot()`, `density()`, `boxplot()`, `parallel_coordinates(missing_column=...)` and
`bar(show_both=True)`.

On `matrix()`, `rate()` and `heatmap()` the same parameter switches the colour bar
instead, which is the only key those plots have. `matrix()` draws its bar as two solid
blocks rather than a gradient, since present against missing has no in-between, and
labels them `NA` and `!NA` to match the legends elsewhere. The bars carry no title at all:
the plot title says what the figure is and the numbers on the bar say what the colours
mean.

`totals()`, `dendrogram()` and `upset()` do not take the parameter, because they label
their marks directly and draw no legend to switch.

### `max_label_length`

Axis labels longer than this are truncated with an ellipsis, then de-duplicated if the
truncation made two labels identical. It defaults to `48`; `0` falls back to a budget
derived from `width`, currently `max(16, width / 12)`.

It exists because column names in real datasets are long, and an axis of rotated
forty-character strings costs more figure area than the plot itself. It was for a while
the parameter that did nothing on four plots — `bar()`, `heatmap()`, `dendrogram()` and
`parallel_coordinates()` drew names at full length however long they were, the heatmap
worst of all since it labels both axes — and all four now truncate as `matrix()`,
`rate()`, `upset()` and `venn()` already did.

The de-duplication is the part worth explaining. Two columns whose names differ only past
the cut come out identical, and plotly treats identical category names as one category,
which merges two real columns into one tick. The disambiguating marker is therefore
visible (`~2`, `~3`) rather than padded whitespace, which would be distinct to plotly
while still reading as a duplicate on screen, and it is taken *out of* the budget rather
than appended on top — otherwise the cap is not a cap, which was a real bug:
`venn(max_label_length=6)` once produced labels of seven and eight characters.

The number 48 itself, and the `width / 12` fallback, are not justified anywhere.

### A note on what is deliberately not a parameter

Several quantities that a plotting library would normally expose are fixed constants
here, each with the reason recorded beside it in the source. Every rate the package
prints uses two decimals, so the same number reads the same way whichever plot drew it;
this used to be a configurable `value_round` with different defaults on different plots
and different hardcoded widths elsewhere. The scatter plot's point size and opacity, the
UpSet dot size, the dendrogram and parallel-coordinates line widths and the density fill
opacity are all set for legibility under overlap rather than to taste. The rate strip's
twenty-column limit for in-cell values is a legibility limit, not a preference, and as an
option it used to silently override an explicit `show_values=True`.

## `matrix()`

One row per observation, one column per variable, each cell coloured by whether that
value is present or missing. This is the plot to start from: horizontal bands mean some
rows are badly affected, vertical bands mean whole columns are unreliable, and a
scattered dusting with no structure is the visual signature of MCAR.

Its column options are the shared ones, with `sort_by` doing considerably more work.

### `sort_by` (extended form)

On `matrix()` alone, `sort_by` accepts a third kind of value: the name of a column in the
DataFrame. The two keywords order the *columns*, as everywhere else; a column name orders
the *rows* by that column's values instead, and pins that column to the left edge so the
ordering it produced can be read off the figure. The y-axis is then labelled with that
column's values rather than with row indices, and the column is outlined in blue as an
annotation.

This exists because sorting rows is how a matrix stops being a texture and starts being
an argument. Gaps that look random in frame order and line up into a band once the rows
are sorted by age are the difference between MCAR and MAR, and this is the parameter that
performs that test.

Rows are ordered with `DataFrame.sort_values`, so the column's dtype decides the order:
numerically for numbers, lexicographically for strings, and by declared category order for
a categorical. That is a deliberate delegation — declaring the order on the column itself
means every plot follows it, rather than each plot growing its own ordering vocabulary:

```python
df["size"] = pd.Categorical(df["size"], categories=["S", "M", "L"])
md.matrix(sort_by="size")  # draws S, then M, then L
```

Missing values sort last whichever direction `ascending` gives. This is not the pandas
default carried over; it is chosen, because reversing a sort should not push the rows
under study — the ones with a missing sort key — off the far end of the figure.

### `sort_categories`

The exact order in which to draw the values of the column `sort_by` names, defaulting to
`None`.

It exists for nominal categoricals, which have no inherent first or last. Sorting them
alphabetically is an arbitrary choice dressed up as a natural one, and this is how the
caller makes it deliberate instead:

```python
md.matrix(sort_by="country", sort_categories=["Portugal", "France", "Spain"])
```

The sequence is drawn exactly as written, so `ascending` does not apply to it: the
sequence already states which value is first and which is last, and honouring a direction
on top of it would draw the reverse of what the caller wrote. Reverse the sequence to
reverse the plot. Values not named follow the ones that are, and missing values come last
— three buckets, always in that order.

Two failure modes are handled differently on purpose. Passing `sort_categories` without a
`sort_by` that names a column raises, because there is nothing sorting the rows for the
order to apply to and accepting it would be indistinguishable from honouring it. Naming
values that the column does not contain only warns, because the same order is worth
reusing across datasets and a value one frame happens to lack is fine — however a typo
looks identical from inside the function, and would otherwise reorder nothing and say
nothing.

It was named `order_categorical` before the API redesign. `sort_categories` joins
`sort_by` and `ascending`, so typing `sort_` in an editor offers the whole mechanism.

### What was removed

`matrix()` used to have a `group_by_mode`. It reversed `z` and the colour scale together,
which cancelled out: the rendered figure was identical either way, missing stayed red, and
the only surviving effect was the order of the two labels on the colour bar. Its one real
consequence was a bug — the hover text was read off `z`, so `group_by_mode="missing"`
labelled every present cell `NA` and every missing cell `!NA`. Hover labels now come from
the missingness mask directly, so they cannot drift out of step with how `z` happens to be
encoded.

`order_by_border_color` and `order_by_border_width` are gone for the reason given above:
the border marks which column `sort_by` named, so it has to read as an annotation, and
nothing is gained by letting it be recoloured into the plot's own palette.

## `bar()`

Missing count per column, which is the plot most people reach for second.

### `measure`

`"count"` (the default), `"fraction"` or `"percentage"`. The count is an absolute number
of rows; the other two are the same number as a share of the dataset, on a 0–1 or a 0–100
scale. The choice of measure also selects which class is built behind the scenes, a
`_BarCount` or a `_BarRate`.

`measure` exists in this form because it absorbed a second parameter. There used to be a
`measure` and a `scale`, and `scale` was ignored whenever `measure="count"` — a state in
which a parameter silently did nothing, which is the pattern the redesign set out to
remove. One option now covers absolute counts and both relative forms, and it means the
same thing on `bar()`, `rate()`, `venn()` and `upset()`.

The default is the count because it is the more concrete number and the one a reader can
act on directly. Rates are the better choice when comparing columns of differing
completeness, which is what the example script says about them.

### `orientation`

`"vertical"` (default) or `"horizontal"`. Horizontal bars keep long column names readable,
since the names run along the axis rather than rotated beneath it. Why vertical is the
default is not recorded; it is the conventional orientation for a bar chart, but nothing
in the repository states that as the reason.

### `show_values`

Draws the numeric value on each bar, `True` by default. The bar height already encodes the
value, so the number is redundant in principle — however it is what makes a figure
quotable, and reading an exact count off a bar against a coarse axis is otherwise
guesswork. Turn it off when the comparison rather than the magnitude is the point.

### `show_both`

Stacks the present count on top of the missing count, so each bar's total height is the
row count and the split is visible at a glance. It defaults to `False` because the missing
count is the whole point of the plot; `show_both` is how a caller asks to see the present
side too. It also changes the axis title, from `Missing rows` to `Rows`, since a stacked
bar means something different from a lone one.

It applies to `measure="count"` only and raises for the rate measures. The reason is
arithmetic: missing and present rates always sum to 1, so every stacked rate bar would be
exactly the same height and the figure would say nothing. Dropping the argument quietly
would draw a plain rate bar that looked as though the stacking had been honoured, which is
why this is an error rather than a no-op.

`show_both=True` is also the one configuration in which `bar()` draws a legend, since it
is the one configuration with two series to distinguish.

### What was removed

`bar()` used to carry `completeness_mode`, `completeness_threshold` and
`max_columns_by_completeness`. Completeness is `1 - missingness`, so all three restated,
inverted, what the shared options already say, and `bar()` was the only plot carrying
them. Removing them is what let `bar()` pick its columns through the same helper as every
other column plot, which in turn removed the last path in the package that drew a blank
chart instead of raising. The migration is
`completeness_mode="at_least", completeness_threshold=0.9` to
`high_missingness_threshold=0.1`; `completeness_mode="at_most"` to
`sort_by="missingness", ascending=False`; and `max_columns_by_completeness=n` to
`sort_by="missingness", max_columns=n`. Two differences are worth watching: the old
threshold compared with `>=` / `<=` where `high_missingness_threshold` keeps on a strict
`<`, so columns sitting exactly on the line change side, and
`max_columns_by_completeness` emitted its survivors in the frame's column order where
`sort_by` emits them sorted.

## `rate()`

The missing rate per column as a single coloured strip: one row of cells, one per column,
shaded by rate. It stays readable on wide datasets where a bar chart would get crowded,
which is the whole reason it exists as a separate plot from `bar(measure="fraction")`.

### `measure`

`"fraction"` (default) or `"percentage"`, and there is deliberately no `"count"`. A rate
strip is the share by definition; a count has no natural colour scale on a single row
because the row length carries no magnitude. The measure also propagates to the colour
bar, which is labelled with a `%` when the cells are.

### `show_values`

Writes the rate into each cell, `True` by default, and is automatically suppressed past 20
columns. The strip is one row, so the value written inside a cell has roughly
`width / n_columns` pixels to live in; past that many columns the numbers collide into an
unreadable smear and colour plus hover carry the reading instead.

The suppression is fixed rather than configurable, and that is the design decision. It
used to be a `max_labels_with_values` option, which silently overrode an explicit
`show_values=True` — whether a number physically fits inside a one-row strip is a
legibility limit, not a preference, and expressing a limit as a preference produces
exactly that kind of contradiction.

### `sort_by`

Defaults to `"missingness"` here rather than to `None`. The strip is read as a ranking, so
frame order would waste it. As noted in the shared section, the reason this default
differs from `bar()` and `matrix()` is not recorded.

## `totals()`

Two bars for the whole DataFrame — present cells against missing cells — with no
per-column breakdown. It takes no column options at all, because there are no columns to
select: this is the one plot in the package that counts the whole grid rather than rows,
and its hover says `cells` for that reason.

### `show_values`

Writes the count and its share of all cells above each bar, `True` by default. Turning it
off leaves the comparison to the bar heights, which is what a slide usually wants; the
numbers are still on the hover.

`totals()` takes no `show_legend`. One trace holds both bars, so a legend entry would read
`trace 0`, and the x-axis already names them. That trace opts out of the legend directly
— it previously set `showlegend=False` only to have the shared layout pass overwrite it,
which is exactly the bug that produced the `trace 0` entry.

## `heatmap()`

Association between columns' missingness. This is the plot that carries the most
plot-specific parameters, because it is really three plots sharing a frame.

### `kind`

`"correlation"` (default), `"direction"` or `"dependence"`. It selects which association
is computed and therefore which class is built.

The three differ in what they read, what they measure it with, and what scale the answer
lands on. Written out once, so that the default is visible rather than implied:

| `kind` | What is compared | Rows used | Measure | Range | Cell is blank when |
|---|---|---|---|---|---|
| **`"correlation"`** *(default)* | one column's missingness against another's | every row | Pearson on two booleans, which is the phi coefficient | −1 … +1 | either column's missingness never varies |
| `"direction"` | one column's *values* against another's missingness | rows where the value column is observed | Pearson between a number and a 0/1 indicator, which is the point-biserial correlation | −1 … +1 | the value column is non-numeric or constant, or the missingness column never varies |
| `"dependence"` *(numeric value column)* | the value column split into two groups by the other column's missingness | rows where the value column is observed | two-sample Kolmogorov-Smirnov statistic, the largest gap between the two groups' distribution functions | 0 … 1 | fewer than two observed values, or the missingness column never varies |
| `"dependence"` *(categorical value column)* | the value column's categories against the other column's missingness | rows where the value column is observed | Cramer's V, which reduces to `sqrt(chi2 / n)` because the missingness side has exactly two levels | 0 … 1 | fewer than two observed values, or the missingness column never varies |

The only mask involved anywhere is `mask_missing`, which is `df.isna()`: `True` where a
cell is empty. `"correlation"` uses it on both axes, and the two value kinds use it on one
axis with the observed values on the other.

Every call below draws the same figure, and we recommend the second form in written
analyses, where naming the kind saves the reader from having to remember which one is the
default:

```python
md.heatmap()
md.heatmap(kind="correlation")
```

`"correlation"` reads the missingness mask alone and shows which columns share a gap
pattern. The other two read the observed *values* of one column against the missingness
of another, which is the distinction the parameter exists for: only a plot that reads
values can separate MAR from MCAR, because knowing that two columns miss together does
not say what opens the gaps.

The two value kinds differ in what they can see and what they can say. `"direction"`
computes the signed point-biserial association, so a cell states whether *higher* or
*lower* values go with the gaps. Carrying a direction is what limits it: a relationship
with no direction is invisible to it. A column whose gaps fall at both of its tails at
once leaves the two groups with the same centre, and the signed statistic reports zero.

`"dependence"` gives that up in exchange for coverage. Each cell is an unsigned distance
from independence on a 0-1 scale, measured by the statistic that suits the value column's
dtype: the two-sample Kolmogorov-Smirnov statistic for numeric columns, Cramer's V for
categorical ones. Both are distances from independence on the same scale, which is what
makes it legitimate for one grid to hold both. Because it never assumes an ordering, it
is also the only kind that reads a nominal column honestly rather than correlating
against whatever integers the categories happen to have been numbered with.

We therefore recommend reading `"dependence"` first to find what is there, then
`"direction"` to ask which way it runs. Reading only the signed kind and stopping at a
flat cell is how a real mechanism gets missed.

The default is `"correlation"` because it is the missingno-equivalent view and the
cheapest to compute.

A third kind, `"predictive"`, was removed. It correlated "column i is present" against
"column j is missing", which is the correlation kind with its sign flipped — presence and
missingness are one mask, so `corr(1 - miss_i, miss_j) == -corr(miss_i, miss_j)` for every
pair. Same grid, opposite poles, no separate information. Passing it still raises a named
error rather than a generic one, because notebooks and scripts written against the
pre-release surface still pass it.

### `show_values`

Writes each association value into its cell, `True` by default, and suppressed
automatically when the matrix is larger than 30 on either side, where the text would be
unreadable. Two formatting decisions sit underneath it: a value that rounds to zero prints
nothing, so the eye is not drawn to a cell that says nothing, and a value that rounds to
exactly ±1 without being ±1 prints `<1` or `>-1`, so a near-perfect association is never
mistaken for a perfect one.

### `drop_constant_columns`

As described in the shared section, defaulting to `False`. On the value kinds it does
something slightly different from the symmetric one, which is worth knowing: it drops
value rows whose association is undefined for every target, and separately drops missing
columns whose missingness has no variance. The two axes mean different things, so
"constant" has to be tested differently on each.

### `show_upper_triangle`

Masks the lower triangle and draws only the upper one, `False` by default.

It exists because a symmetric matrix says everything twice, and half a symmetric matrix is
the same information in half the area. We default it off because the full square is the
more familiar shape and reads faster at small sizes; the mask is the option you reach for
when the matrix is large enough that the duplication costs real space.

It applies to `kind="correlation"` only and raises for the value kinds. Their matrix
says nothing twice: its axes carry different meanings, values on one and missingness on
the other, so `value(a)` against `missing(b)` and `value(b)` against `missing(a)` are
different questions. Masking by position there would delete real associations rather than
duplicates.

### `selected_value_columns` and `selected_missing_columns`

Two separate column selections for the value kinds: the first names the columns whose
*values* form one axis, the second names the columns whose *missingness* forms the other.
Each falls back to `selected_columns` when not given.

They exist because the value matrix is the only asymmetric one in the package, and a
single `selected_columns` cannot express "do age and income explain the gaps in score and
rating". Passing either for another kind raises, since on a symmetric matrix the two axes
are the same list and the parameters would have no meaning to honour.

For the reader, these are what turn the heatmap from a survey into a specific hypothesis:
a narrow rectangle of value columns against missing columns is a question, where the full
square is a fishing expedition.

### `sort_by`

Defaults to `"missingness"`, as on `rate()`. On the value kinds it orders both axes, since
both are column lists, and it is read as a value rather than as a flag: `"alphabetical"`
sorts alphabetically there exactly as it does on the symmetric kind. An unrecognised key
raises, matching the shared selection helper that the symmetric kind routes through.

### `max_columns`

Caps both axes on the value kinds, not just one, since both are column lists.

### The colour scale follows the statistic

The signed kinds run from `present_color` through white to `missing_color` across -1 to 1,
with white at a meaningful zero. The unsigned kind runs from white to `missing_color`
across 0 to 1 instead. This is not a cosmetic difference: a diverging bar on a statistic
that cannot go negative would invent a half that never occurs and would put independence
in the middle of the scale rather than at its end, which is where a reader looks for it.

### What was removed

`present_missing_corr` and `present_present_corr` were removed as exact duplicates of
other metrics, in the same spirit as `kind="predictive"`: the missingness mask and the
presence mask are one fact, so a metric computed from either is the same metric.

`kind="biserial"` was renamed rather than removed. It named the statistic, and the same
grid now offers two, so the name no longer said which one a caller would get. It raises
with the new name rather than aliasing silently.

## `dendrogram()`

Hierarchical clustering of columns by missingness correlation, with distance defined as
`1 - correlation` between the columns' missingness masks. Columns joined low in the tree
tend to go missing together. It groups the heatmap's pairwise view into nested clusters,
which is easier to read once there are more than a handful of columns.

It takes `selected_columns`, `high_missingness_threshold`, `max_columns` and
`drop_constant_columns`, but deliberately no `sort_by` or `ascending`: leaf order is
produced by the clustering, and imposing an order on top of it would break the tree.

### `linkage`

The linkage criterion passed straight to `scipy.cluster.hierarchy.linkage`, choosing how
the distance between two clusters is measured. It accepts scipy's seven methods —
`"single"`, `"complete"`, `"average"`, `"weighted"`, `"centroid"`, `"median"` and
`"ward"` — and defaults to `"average"`.

It exists because the linkage choice genuinely changes the tree, and a clustering plot
that hides it is asking to be trusted more than it should be.

The default is `"average"` because that is missingno's default: its `dendrogram` takes
`method='average'`, and this package is deliberately aligned with missingno so that the
same DataFrame produces a recognisably similar figure through either. We have confirmed
this against the installed missingno rather than inferring it. Average linkage is also
the conventional choice for a distance derived from a correlation, since it depends on
every pairwise distance between two clusters rather than on the single closest or
furthest pair, which makes it less sensitive to one unusual column than `"single"` or
`"complete"`.

The parameter was called `linkage_method` before the redesign. It was the package's only
`_method` suffix, and scipy's own argument is just `method`, so the suffix was dropped.

### `use_abs_correlation`

Clusters on the absolute correlation, `False` by default, so that a strongly negative
relationship counts as close rather than far apart.

It exists because two columns that are *never* missing together are, for the purpose of
understanding a missingness mechanism, just as related as two that always are — the
example script makes exactly this argument. However the sign carries real information, so
folding it away has to be an explicit request rather than the default. The `use_` prefix
says the flag changes how something is computed rather than how it is drawn.

Two presentation choices around the tree are fixed rather than exposed. The lines are
drawn heavy because the tree *is* the plot, structure rather than annotation, and they are
drawn in a neutral blue because the tree carries no missing-or-present meaning and must
stay out of the package's red-and-green vocabulary. Both axes are padded, since scipy
places leaves half a step from the data edge and the outermost leaves otherwise sat flush
against the frame.

## `venn()`

The seven exclusive missingness regions of three columns, drawn as bars: rows missing only
A, only B, both A and B, and so on.

### `selected_columns`

Required, and it must name exactly three columns. This is the strictest requirement in the
package and it follows from the plot's definition: a three-set Venn has seven exclusive
regions, and no other number of columns produces that figure. Choosing the three
ourselves — by missing rate, say — would mean the figure silently answered a different
question from the one asked. Naming more than three raises and points at `upset()`, which
is the plot that scales.

### `measure`

`"count"` (default), `"fraction"` or `"percentage"`, meaning exactly what it means on
`bar()`. A share is the comparable form when reporting one dataset alongside another,
which is what the example script uses it for. Regardless of which measure is set, the
hover shows the count *and* the share, on the reasoning that the tooltip has room for
both and should never make the reader change a parameter to see the other number.

### `show_values`

Draws the value on each bar, `True` by default. Zero-sized regions print nothing rather
than a `0`, since an empty region is more clearly read as an empty bar.

### `sort_by` and `ascending`

`sort_by` accepts `"size"` (the default) or `None`, ordering the regions by how many rows
they cover rather than ordering columns. In the current implementation only `ascending` is
consulted: the regions are emitted in enumeration order unless `ascending=True`, in which
case they are sorted smallest first. The documented behaviour of `sort_by="size",
ascending=False` — largest first — is not what the code does, and the default order is the
Venn enumeration rather than a size ranking.

`venn()` takes a `show_legend` parameter but draws no legend, since it has only one
series.

## `upset()`

Every missingness intersection across any number of columns, with bars for the
intersection sizes, a horizontal panel for the per-column totals, and a dot matrix saying
which columns each intersection covers. Use `venn()` for exactly three columns and this
for more.

### `selected_columns`

Required, and every named column is drawn. There is deliberately no cap on the number of
sets: UpSet exists precisely to compare more columns than a Venn can, so capping the sets
would defeat the point. The intersection cap does the legibility work instead.

Like `venn()`, the error raised when it is omitted lists the columns worth comparing, so
requiring it costs the caller nothing.

### `measure`

`"count"` (default), `"fraction"` or `"percentage"`, again the same vocabulary. What is
specific here is that it scales *both* bar panels, so intersection sizes and per-column
set sizes stay on one scale and remain comparable against each other. The axis titles
change with it — `Rows`, `Fraction of rows`, `Percent of rows` — because nothing labelled
them before, which left `4` and `0.20` looking alike.

### `show_values`

Draws the size on each intersection bar, `True` by default.

### `sort_by` and `ascending`

As on `venn()`, `sort_by` takes `"size"` or `None` and orders intersections rather than
columns. In the current implementation the intersections arrive already sorted largest
first and only `ascending=True` changes that; `sort_by=None` does not restore an
enumeration order, contrary to the docstring.

### The intersection cap

Not a parameter, and that is the decision. Intersections grow as `2**n`, so the bar count
has to be capped or a wide selection draws an unreadable comb; the cap is fixed at 20,
largest first. A selection producing more warns and names how many were found, rather than
truncating quietly, because drawing the top 20 of 300 without saying so reads as "these
are all the patterns", which is the opposite of what the plot is for. The empty
intersection is dropped and every intersection that occurs at all is drawn, so the size
floor is 1: on this plot a pattern that happens once is still a pattern.

Note that `docs/plots.md` currently mentions a `max_intersections` parameter. No such
parameter exists.

## `scatterplot()`

A scatter of two columns that keeps missing values visible. A normal scatter drops any row
where either axis is missing; here those rows are offset outside the data range with a
distinct marker, so the values that cannot be plotted stay visible and countable. Marker
shape carries which axis is missing: `x` for a missing x, a down-triangle for a missing y,
an open diamond for both.

### `x` and `y`

The two columns, positional and required. `scatterplot()` keeps the names `x` and `y`
where `density()` and `boxplot()` were renamed away from them, because this is the one
plot that genuinely has both an x and a y.

### `axis_padding`

The fraction of the data span added as padding at each end of both axes, `0.1` by default.
When `missing_column` is set, the padding above the data is widened to `0.15` to leave
headroom for the legend.

It exists to give the offset band room to breathe. The offset position itself is *not*
governed by this parameter: missing values are parked a fixed tenth of the observed span
below the minimum, and the computed range always includes that position, so
`axis_padding=0` still shows the markers — flush against the frame rather than off it.
The method docstring currently states that the padding is where the offset markers are
placed, which overstates the parameter's role.

Why `0.1` specifically is not recorded.

### `missing_column`

Colours points by whether a *third* column is missing in that row, instead of by the
missingness of `x` and `y`. It defaults to `None`, meaning the points are coloured by
their own axes' missingness.

It exists because the interesting question is often not "which of these two values is
missing" but "does the relationship between these two values explain a gap somewhere
else". The marker shapes still encode x-and-y missingness when it is set, so the two
readings coexist rather than replace each other. It raises if the column is not in the
DataFrame.

It was called `missingness_color_column`. That name sat one suffix away from
`missing_color` while meaning something else entirely — a column name, not a colour.

### `jitter`

Gaussian noise added to every plotted point, as a fraction of each axis span, defaulting
to `0.02`. Setting it to `0` gives exact positions at the cost of hiding density.

It exists so that coincident values separate instead of stacking into a single mark, which
on a dataset with repeated or rounded values is the difference between "one observation"
and "four hundred". The offset markers need it most of all: every missing row sits on
exactly the same coordinate and would otherwise draw as one point however many rows it
stands for.

The offset band's jitter is capped separately, at a third of the offset gap, so that
roughly three standard deviations still fit inside the gap. Without the cap a large
`jitter` would push missing points up into the observed range, where nothing would tell
them apart from real values — the plot would then be lying about data it exists to be
honest about.

The value `0.02` itself is not justified in the repository.

### `jitter_seed`

The seed for the jitter, `42` by default, so that a figure is reproducible. It exists
because jitter is randomness added to a figure that may end up in a paper, and a figure
that redraws differently every run cannot be cited. The particular value is the usual
convention and no reason for it is recorded.

### `xaxis_range` and `yaxis_range`

Explicit `[min, max]` overrides for the computed axis ranges, `None` by default. They
exist for the ordinary reasons — comparing two figures on one scale, or cutting an outlier
that flattens everything else. The computed default is what you want almost always, since
it is the only range guaranteed to contain both the data and the offset markers; an
explicit range that excludes the offset band will hide the missing values the plot is for.

## `density()`

Two overlapping KDE curves for one column, split by whether another column is missing.
Curves that sit on top of each other are consistent with MCAR; curves that pull apart mean
the distribution changes with missingness, which is MAR or MNAR.

### `column` and `missing_column`

Both positional and required. `column` is the numeric column whose distribution is
estimated; `missing_column` is the column whose missingness splits the rows into the two
curves. The plot raises a `TypeError` naming the dtype if `column` is not numeric.

`column` was called `x`. The rename is deliberate: neither `density()` nor `boxplot()` has
a `y`, and on `boxplot()` the column was drawn on the *vertical* axis, so `x` described a
position it did not occupy.

`density()` has no other plot-specific parameters, which is itself a decision — the
options a KDE plot would normally expose are fixed. The curve is sampled at 300 points,
which sets resolution and not shape; above a few hundred points a denser grid only makes
the figure heavier. The fill opacity is low enough that the two curves stay
distinguishable where they overlap. A group with no spread gives a singular covariance
matrix that a Gaussian KDE cannot invert, so that group falls back to a histogram rather
than raising.

The y-axis deliberately prints no numbers. A KDE integrates to 1 over the column's range,
so on a wide axis the values land near `1e-5` and plotly renders them as `60u`, which
carries nothing for a reader; the shape and the point where the curves separate is what
the plot is read for. Because of that, the hover is the only place a reader can find how
much data sits under each curve, which is why it always carries the group's row count
against the total.

## `boxplot()`

The same comparison as `density()`, drawn as boxes or violins: the distribution of one
column split by whether another is missing.

### `column` and `missing_column`

As on `density()`, positional and required, with `column` on the vertical axis and
`missing_column` supplying the split. A non-numeric `column` raises a `TypeError` that
suggests `pd.factorize` as the fix rather than merely reporting the dtype.

### `kind`

`"box"` (default) or `"violin"`. Boxes show medians and quartiles and are easier to
compare across groups; violins additionally show the shape of each distribution. The
default is the simpler of the two, and the docstring frames it that way, though no
stronger justification for the choice is recorded.

It was called `plot_type`. `kind` is the package's word for "which variant of this plot",
whereas `plot_type` only restated the method name.

A box hides its own sample size, which is the first thing to check before reading anything
into a spread, so the hover carries the group's row count against the total.

## `parallel_coordinates()`

Every row drawn as a line crossing all the axes, coloured by one column's missingness. It
is the plot that shows all columns at once, so a multivariate pattern behind the gaps
appears as the two colours separating.

### `missing_column`

The column whose missingness colours each line, `None` by default. Without it every line
is the same colour and no legend is drawn, since a legend entry would name a distinction
that is not drawn — which makes `missing_column` effectively the parameter that turns this
into a diagnostic plot rather than a survey.

Note that the method docstring still claims the legend is titled `Status` when the
parameter is omitted and `"<column>_NA"` otherwise. Neither is true of the current code:
`legend_title` was removed package-wide and the entries are now spelled `NA-<column>` and
`!NA-<column>`, matching the vocabulary every other plot uses.

### `kind`

`"values"` (default) or `"missingness"`. With `"values"` each column is normalised onto a
shared 0–1 scale, so a line traces a row's actual measurements; with `"missingness"` every
column is drawn as a binary present-or-missing instead.

The second is explicitly the escape hatch for non-numeric columns, which cannot be
normalised onto a shared axis at all — `kind="values"` raises a `TypeError` naming them.
The default is `"values"` because reading the measurements is the richer view when it is
available.

This was a boolean called `missingness_only`. It selects a variant, so it belongs on
`kind` rather than being the one boolean in the package with no prefix.

### `sort_by`

The shared parameter, however it carries more weight here than on any other plot, and that
is worth stating explicitly: a relationship between two columns is visible only where
their axes are *adjacent*. Ordering by missingness puts the columns most likely to share
gaps next to each other, which is the difference between seeing a pattern and not. On a
bar chart the order is a convenience; here it decides what the figure can show.

Because the normalised y values are unreadable as numbers, the hover carries the raw value
for each vertex. That is the whole reason this plot needs a tooltip.

## `Panel`

Combines several plots into one grid. It is not a plot factory but a composition object,
so its parameters are about layout rather than about data.

### `plots`

The list of plot objects to combine, optional because plots can also be added one at a
time with `add()`, which returns the panel for chaining. The plots are deep-copied into
the combined figure, so the originals are not mutated and can still be shown or saved on
their own.

### `title` and `description`

The panel-level title, drawn above the whole grid, and an optional subtitle drawn beneath
it. Each sub-plot keeps its own `title`, which becomes its subplot title — which is why
the documentation recommends giving every plot in a panel a title of its own. The default
title is `"Combined Plots"`, and the download filename falls back to `panel` rather than
slugifying that placeholder. No rationale for the `description` parameter is recorded
beyond its being a subtitle.

### `max_cols` and `max_plots`

The maximum number of grid columns, `3`, and the maximum number of plots a panel accepts,
`6`. Exceeding `max_plots` raises rather than silently dropping the extras, both on `add()`
and when the figure is built.

Neither number is justified anywhere in the repository. Three columns and six plots is a
readable screen, and a cap that raises is consistent with the package's treatment of
truncation elsewhere, however nothing in the code or the history states either as the
reason.

### `width` and `height`

Both `None` by default, in which case the panel sizes itself: 1500 pixels wide for three
columns, 1400 for two, 800 for one, and `450 * rows + 150` tall. The heuristic exists
because a panel's natural size depends on its grid rather than on any single plot, so
carrying the sub-plots' own `width` and `height` through would produce overlap. The
particular figures are not justified.

### `background_color` and `text_color`

As on the individual plots, applied to the whole grid. Colour bars and legends are hidden
inside a panel regardless of what the individual plots requested, since at panel scale
they overlap the neighbouring cells and the sub-plot titles carry the identification
instead.

### `save(save_individual=True)`

Not a constructor parameter, however it belongs here: saving a panel with
`save_individual=True` additionally writes each sub-plot as its own PNG into the same
directory, named from the plot's class and title. It exists because the panel is usually
the overview and the individual figures are what end up in a document.

## Where the constants come from

Ten defaults were, at one point, documented here as having no recorded reason. Most of
them turned out to have one that is recoverable by measurement rather than by memory, and
the measurements are recorded below so that the next person to question a number has
something to argue with. The remainder are conventions, and we say so.

All of these live in the source rather than in any notebook. The overridable ones are
signature defaults in `core/mixins/_plots.py` and on `Panel`; the rest are either
keyword-only arguments on the private `_Plot._truncate_labels`, or module-level constants
in the individual plot files, where a caller cannot reach them at all.

### `width=900` and `height=520`

The published documentation renders inside mkdocs-material's content column, which is
`61rem` wide with `1.2rem` of padding on each side: 976 pixels, of which 937.6 are usable
at the default 16-pixel root. A 900-pixel figure is therefore the largest round number
that renders without a horizontal scrollbar on this project's own site. 520 gives a
landscape shape close to 16:9, which would be 506 at that width.

The caps of 2000 and 1000 remain unjustified beyond being roughly twice the defaults.
Their *existence* is defended in the code, since an unbounded figure size is a way to
produce an unopenable HTML file; the two numbers are not.

### `max_label_length=48`, and the `width / 12` fallback

These are calibrated to the width of a character, and the arithmetic is exact enough to be
worth stating. Measured in a 12-pixel sans-serif face, 48 wide characters occupy 561
pixels, which is 11.69 pixels each. The `width_divisor` of 12 is that measurement rounded
up: at the default width it yields `900 / 12 = 75` characters, and 75 characters at 11.69
pixels is 877 pixels, or essentially the entire figure. The fallback budget is therefore
the point at which a single label would span the whole plot.

`max_label_length=48` sits deliberately below that geometric maximum, leaving a label at
roughly 62 per cent of the figure width, because a label that reaches the far edge is
technically drawn and practically unreadable.

The `min_len=16` floor binds only below `16 * 12 = 192` pixels of figure width, which is
narrower than any figure this package will draw by default. It exists so that the budget
cannot collapse to nothing on a deliberately tiny figure.

### The 30-cell threshold on `heatmap(show_values=)`

This one is exact. At the default width of 900, a heatmap of 30 columns gives each cell
`900 / 30 = 30.0` pixels. A typical in-cell value, `-0.74`, measures 31 pixels in the same
12-pixel face. The threshold sits precisely at the crossover: at 30 columns the widest
common cell text no longer fits, at 20 columns it fits with margin (45 pixels of cell for
31 pixels of text), and at 40 it overflows by nearly 40 per cent.

We therefore suppress the numbers above 30 rather than draw text that overlaps its
neighbours. Note that the threshold is a fixed count rather than a function of `width`,
so a deliberately wide heatmap still loses its numbers at 31 columns. That is a
simplification, not a measurement.

### `jitter=0.02` and `axis_padding=0.1` on `scatterplot()`

Both are tied to `_OFFSET_GAP`, the fixed tenth of the observed span at which missing
values are parked outside the data.

`axis_padding` defaults to `0.1`, which is exactly `_OFFSET_GAP`. The frame is therefore
padded by one offset-gap, which places the offset markers one full gap inside the plot
edge rather than flush against it.

`jitter` defaults to `0.02`, against a cap of `_OFFSET_GAP / 3 = 0.0333` applied to points
in the offset bands. The default is 60 per cent of that cap, and 20 per cent of the offset
itself: enough to separate coincident points, and comfortably short of the distance that
would let a jittered offset marker be mistaken for an observed value.

### `Panel` sizing

The heuristic is easier to read as per-sub-plot arithmetic than as the totals it is
written in. A three-column panel is 1500 pixels for 500 each, two columns is 1400 for 700
each, and one column is 800. Height is `450 * rows + 150`, where the 150 is the space the
panel title and its description occupy above the grid.

`max_plots=6` is `max_cols=3` by two rows, which is the largest panel that stays under
1050 pixels tall. Panels are consistently wider than the 937.6-pixel documentation column
and are scaled down by the browser, which is why sub-plots inside a panel drop their
colour bars and legends.

### Conventions, not measurements

Four defaults are conventions and we record them as such rather than dress them up.
`bar(orientation="vertical")` is the conventional orientation for a bar chart, and the
horizontal form exists for long category names. `boxplot(shape="box")` is the more widely
read of the two shapes. `show_values=True` reflects a general preference in this package
for figures that can be read without hovering, with the automatic suppression thresholds
above doing the work of keeping that honest. `jitter_seed=42` is a convention for a
reproducible seed and nothing more.

### Still unexplained

Two entries survive. The `2000` and `1000` size caps are arbitrary numbers around an idea
that is not. And the 30-column threshold, while measured against the default width, does
not adapt to a width the caller changes.

Three earlier entries were closed by finding a fact rather than inventing a reason: the
default colours are `tab10[3]` and `tab10[2]` from matplotlib's cycle, verified against
matplotlib; the default linkage is `"average"` because missingno's `dendrogram` defaults
to `method='average'`, verified against the installed missingno; and the `sort_by`
discrepancy on the value heatmaps turned out to be a defect rather than a decision, and
has been fixed.
