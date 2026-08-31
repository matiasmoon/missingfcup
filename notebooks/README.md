# Notebooks

Six end-to-end analyses that take a real dataset, inject missing values of a known
mechanism, and use `missingfcup` to recover that mechanism from the data alone. Because
the answer is known in advance, each notebook is a check on what the plots *mean*, not
only a gallery of what they look like.

**Read them here: [Analyses](https://matiasmoon.github.io/missingfcup/notebooks/).** The
documentation site serves them as web pages with the figures already executed and still
interactive. The `.ipynb` preview on github.com shows nothing, because it strips the
JavaScript the figures depend on.

**Run them yourself:**

```bash
pip install -e ".[notebooks]"
jupyter notebook notebooks/titanic/analysis_multi.ipynb
```

New to these? Skip to
[How to read these notebooks](#how-to-read-these-notebooks).

## Structure

The `examples/` folder shows one plot per script and the test suite asserts figure
content, but neither exercises the package the way an analyst does. These notebooks are
the only place where the whole workflow runs end to end.

Each dataset has its own folder containing a generation notebook, an analysis notebook,
the generated CSV files, and the figures the analysis writes.

```
<dataset>/
  gen_multi.ipynb        creates the CSVs; needs network access
  analysis_multi.ipynb   reads the CSVs and does the analysis
  data/                  mcar_multi.csv, mar_multi.csv, mnar_multi.csv (committed)
  plots/                 PNGs written by the analysis (git-ignored)
```

The CSVs are committed, so an analysis runs offline and reproduces exactly what is shown.

`mechanism_variants/` holds the same four files but is not a dataset: it varies the
*generator* while holding the data fixed, so it sits alongside the dataset folders rather
than inside one of them.

## Datasets

Five datasets, chosen to stress different shapes rather than to repeat one example.

| Dataset | Source | Rows | Columns | Shape |
|---|---|---|---|---|
| `titanic` | seaborn | 712 | 5 | numeric only; the only one with both a univariate and a multivariate analysis |
| `breast_cancer` | scikit-learn | 569 | 30 | all continuous |
| `contraceptive_method` | UCI (id 30) | 1473 | 9 | categorical, factorized to integer codes |
| `student_performance` | UCI (id 320) | 649 | 30 | mixed types, factorized to integer codes |
| `default_credit` | UCI (id 350) | 30000 | 23 | anonymised `X1..X23` column names |

## Mechanisms

Missingness is injected with [`mdatagen`](https://github.com/ArthurMangussi/pymdatagen)
under a fixed seed of 42, reseeded immediately before each generator so that every
mechanism draws from the same starting point and is independent of what the others
consumed. The parameters differ by dataset:

| Dataset | MCAR | MAR | MNAR |
|---|---|---|---|
| `titanic` (uni) | `uMCAR`, 20% on `age` | `uMAR.highest`, `x_obs="pclass"` | `uMNAR.run`, 20% on `age` |
| `titanic` (multi) | `mMCAR.random`, 20% | `mMAR.correlated`, `n_xmiss=3` | `mMNAR.MBOV_randomness` on `age`, `fare` |
| `breast_cancer` | `mMCAR.random`, 30% | `mMAR.random`, `n_xmiss=15` | `mMNAR.random`, `threshold=1`, `n_xmiss=15` |
| `contraceptive_method` | `mMCAR.random`, 30% | `mMAR.random`, `n_xmiss=4` | `mMNAR.random`, `threshold=1`, `n_xmiss=4` |
| `student_performance` | `mMCAR.random`, 30% | `mMAR.random`, `n_xmiss=15` | `mMNAR.random`, `threshold=1`, `n_xmiss=15` |
| `default_credit` | `mMCAR.random`, 30% | `mMAR.random`, `n_xmiss=15` | `mMNAR.random`, `threshold=1`, `n_xmiss=15` |

The MNAR generators drive missingness from the prediction target, which is not part of
the feature matrix. Its effect can therefore only be seen through proxy columns, which
is what makes the MNAR sections the hardest of the three to argue.

One detail matters when reading the CSVs: `mdatagen` appends a `target` column to the
MAR and MNAR outputs but not to the MCAR one. The analysis notebooks drop it before
constructing `MissingData`, so that all three mechanisms are compared over the same
feature columns.

## How to read these notebooks

Start with `titanic/analysis_multi.ipynb`. It is the smallest dataset, five numeric
columns, and every plot in it is readable without scrolling. The other four repeat the
same method on harder data, so once the titanic notebook makes sense the rest are
variations on a template you already know.

### The three files inside one notebook

Each analysis loads three CSV files from the same dataset: `mcar_multi.csv`,
`mar_multi.csv` and `mnar_multi.csv`. They hold identical rows and columns and differ
only in which cells were emptied. Every plot is therefore drawn three times, once per
mechanism, and the interesting content is the difference between the three panels rather
than any one of them on its own.

### The four sections

1. **Exploratory Discovery** — how much is missing and where. `totals`, `bar` and `rate`
   give the amounts, `upset` and `venn` show which columns go empty in the same rows.
   Nothing here separates the mechanisms; it establishes what the later sections are
   working with.
2. **Mechanism Discovery** — the core of the notebook. `matrix`, the three `heatmap`
   kinds, `dendrogram` and `parallel_coordinates` look for structure in the gaps.
3. **Comparative Discovery** — `density`, `boxplot` and `scatterplot` ask whether the
   *values* of a column differ between the rows where another column is present and the
   rows where it is missing.
4. **Conclusions** — the verdict for each mechanism, with the evidence that supports it.

Sections 2 and 3 are where the argument is made, and they attack the problem from two
directions. Section 2 reads only the missingness mask: which cells are empty, ignoring
what the filled cells contain. Section 3 reads the values. Some mechanisms are invisible
to one and obvious to the other, which is why both are there.

### What each plot answers

| Plot | Question it answers |
|---|---|
| `matrix` | where do the gaps sit, row by row? |
| `matrix(sort_by=...)` | do the gaps line up when rows are ordered by a candidate driver? |
| `heatmap()` | which columns go missing *together*? |
| `heatmap(kind="direction")` | do a column's *values* predict another column's gaps, and which way? |
| `heatmap(kind="dependence")` | the same, unsigned, so gaps at both tails still register |
| `dendrogram` | the same as `heatmap()`, grouped into clusters instead of pairs |
| `parallel_coordinates` | do the missing rows sit in a particular part of the value space? |
| `density`, `boxplot` | does this column's distribution shift when another column is missing? |
| `scatterplot` | the same question for two columns at once, keeping the unplottable rows visible |

### The decisive move

The single most important plot is `matrix(sort_by=driver)`: the row-by-row grid with the
rows reordered by the values of a candidate driver column.

If the driver explains the gaps, sorting by it collects them into one contiguous block.
If it does not, they stay scattered. That block is the clearest signal any of these
notebooks produce, and each one uses it to separate the three mechanisms:

* **MCAR** — no column produces a block, and every other plot is flat as well.
* **MAR** — one or more columns produce a block, and the direction heatmap names the same
  driver independently.
* **MNAR** — no observed column produces a block, yet the distributions are visibly
  distorted. The driver exists but is not in the data, so it can only be seen by its
  effect on the values that remain.

MNAR is therefore argued by elimination plus distortion, never by a positive block, and
it is the hardest of the three to demonstrate.

### Reading the numbers

Every dataset analysis closes with `littles_mcar_test()` and `mann_whitney_test()`, which
put a p-value on conclusions the plots reach visually.
(`mechanism_variants/analysis_variants.ipynb` uses Little's test only, since it compares
generators rather than columns.)

Do not treat Little's p as the verdict. Across these five datasets it misleads on three
of them:

| Dataset | Little's p on the MCAR data | Reading |
|---|---|---|
| `titanic` (both) | 0.62, 0.34 | correct, MCAR not rejected |
| `student_performance` | 0.24 | correct |
| `contraceptive_method` | 0.039 | **false positive** at the 0.05 threshold |
| `default_credit` | 0.0 | **false positive**; 30000 rows gives it power to flag trivia |
| `breast_cancer` | 1.0 (and 1.0 for MAR) | **uninformative**; 30 columns over 569 rows degenerates the statistic |

`perfectly_correlated_missing_columns()` is the dependable one: it returns identical
pairs under MAR in every dataset and none under MCAR or MNAR. `missing_pattern_counts()`
points the same way from the row side, falling under MAR everywhere — 30 patterns to 4 on
`titanic`, 357 to 8 on `contraceptive_method`, 649 to 221 on `student_performance`, 569 to
183 on `breast_cancer`, 28699 to 565 on `default_credit` — but it is the weaker signal,
because MNAR collapses too and on `titanic` lands on the same count of 4 that MAR does.

## Where each feature is demonstrated

Every multivariate dataset analysis exercises the whole package. This table is for
finding a specific feature, and for seeing where one is deliberately absent.
`mechanism_variants/` is listed only where it earns a row: it answers a narrower question
with three plots, `matrix`, `heatmap` and `density`, and leans on the metrics instead.

| Feature | Where |
|---|---|
| all 12 plots | every multivariate dataset analysis |
| `heatmap(kind="direction")` | every dataset analysis |
| `heatmap(kind="dependence")` | `contraceptive_method`, whose columns are categories stored as codes |
| `parallel_coordinates(kind="missingness")` | every multivariate dataset analysis |
| `boxplot(shape="violin")`, `bar(measure="fraction")` | every dataset analysis |
| `max_columns=` | the four wide datasets: on `matrix` in three, on `parallel_coordinates` in `student_performance` |
| `littles_mcar_test()`, `mann_whitney_test()` | every dataset analysis |
| `ks_test()` | `contraceptive_method` and `mechanism_variants`, beside the reading it disagrees with |
| `perfectly_correlated_missing_columns()`, `missing_pattern_counts()` | every multivariate dataset analysis |
| metrics (`total_missing_rate`, `col_missing_rate`, `rows_complete`, …) | every dataset analysis |

Two absences are deliberate. `titanic/analysis_uni.ipynb` has one incomplete column, so
everything that compares columns against each other — `upset`, `venn`, `dendrogram`, the
correlation heatmap, both pattern functions, the missingness mode of
`parallel_coordinates` — has nothing to work with. It closes with a section saying so,
which makes it the one notebook here that shows when *not* to reach for a plot. And
`titanic/analysis_multi.ipynb` skips `max_columns`, since five columns need no cap.

## Mechanism variants

The five dataset analyses each validate one `mdatagen` method per mechanism, which leaves
a question none of them can answer: is the success a property of the mechanism, or of the
particular implementation that produced it?

`mechanism_variants/` applies all 21 methods `mdatagen` exposes to `titanic` and
`breast_cancer` — a narrow mostly-continuous dataset and a wide fully-continuous one — and
asks what the package can tell apart. The two categorical datasets are excluded on
purpose: most of the API selects rows by where a value sits in its column's order, and
those datasets are `pandas.factorize` output whose integer codes carry no order, so those
methods would run and return artefacts. `default_credit` is excluded on size.

Three findings come out of it:

* **Mask-based diagnostics separate mechanism classes, not implementations.** 21 methods
  produce only 10 distinct structural fingerprints on `titanic` and 11 on `breast_cancer`.
  All eight univariate methods collapse into one, since a single incomplete column admits
  exactly two row patterns however the rows were chosen.
* **The value-based views are what recover the difference.** `heatmap(kind="direction")`
  and the distribution plots separate `uMAR.highest` from `uMAR.lowest` cleanly, where
  every mask-based plot shows an identical picture.
* **`mMNAR.median` defeats the whole toolkit on narrow data.** On `titanic` it is
  indistinguishable from `mMCAR.random` by every diagnostic available, Little's test
  included (p=0.482 against 0.618). It is caught on `breast_cancer`, but by a test that
  returns `nan` for the MCAR baseline on that same dataset.

Three `mdatagen` behaviours worth knowing are recorded there as well. `mMNAR.MBIR`
overshoots its requested rate, delivering 0.600 against the 0.200 asked for on `titanic`
and leaving zero complete rows. `mMAR.pattern_missingness` undershoots on both datasets,
delivering 0.208 and 0.227 within the affected columns and only 0.083 and 0.113
dataset-wide. And `uMAR.rank()` raises `KeyError` unless the index is reset to `0..n-1`,
which the gapped index left by `dropna()` does not satisfy.

## Running them

Run a `gen_*.ipynb` only if you deleted the CSVs or changed a generator parameter. Those
notebooks need one more extra, and a newer Python:

```bash
pip install -e ".[notebooks,generate]"
```

That extra holds `mdatagen`, which injects the missingness, and the three libraries the
generators use to download their source data: `seaborn` for titanic, `scikit-learn` for
breast_cancer, and `ucimlrepo` for the three UCI sets. No analysis notebook imports any
of them, which is why they are not in `notebooks`.

It also needs a newer Python. `mdatagen` requires 3.10.12 or above, so `[generate]` will
not install on 3.9 — everything else in this folder runs on 3.9, the package floor. And
the generators download their source dataset, so they need network access.

The analysis notebooks write PNGs into `plots/`, which requires a browser for image
export. `plotly_get_chrome -y` fetches one if `kaleido` has not already done so.

`default_credit` is the one notebook that renders static images rather than interactive
figures. Its 30000 rows make a single interactive matrix roughly 14.5 MB of JSON, and
keeping all of its figures interactive produced a 130 MB notebook — past the limit GitHub
accepts and past what a browser can open. It therefore sets
`pio.renderers.default = "png"` where the others set `"notebook"`.

`make notebook-check` executes `titanic/analysis_multi.ipynb` against the committed CSVs
without writing anything back. CI runs the same target, which is what keeps these
notebooks from drifting out of step with the package API.

`make docs` stages these notebooks into `docs/` and builds the site around them. The
staged copy under `docs/notebooks/` is generated and git-ignored; the originals here are
the source. Because the site renders the committed outputs rather than re-running the
notebooks, re-run them and commit the result whenever the analyses change, or the site
will keep showing the previous figures.
