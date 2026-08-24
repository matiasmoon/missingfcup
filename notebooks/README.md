# Notebooks

This folder holds the end-to-end analyses that `missingfcup` was built for. The
`examples/` folder shows one plot per script and the test suite asserts figure content,
but neither exercises the package the way an analyst does. These notebooks do: they take
a real dataset, inject missingness of a known mechanism, and then use the package to
recover that mechanism from the data alone.

These notebooks are published as part of the documentation site, already executed, under
**[Analyses](https://matiasmoon.github.io/missingfcup/notebooks/)**. That is the place to
read them: the site serves them as ordinary web pages, so the Plotly figures keep their
hover text and zoom. The `.ipynb` preview on github.com cannot show them at all, because
it strips the JavaScript the figures depend on.

`default_credit` is the one exception, and it renders as static images. Its 30000 rows
make a single interactive matrix roughly 14.5 MB of JSON; keeping all of its figures
interactive produced a 130 MB notebook, which is past the limit GitHub accepts and far
past what a browser can open. It therefore sets `pio.renderers.default = "png"` where the
others set `"notebook"`.

Because the missingness is injected rather than found, the correct answer is known in
advance. Every claim a notebook makes can therefore be checked against the generator
that produced the gaps, which makes this folder a validation of what the plots mean and
not only a gallery of what they look like.

## Structure

Each dataset has its own folder containing a generation notebook, an analysis notebook,
the generated CSV files, and the figures the analysis writes.

```
<dataset>/
  gen_multi.ipynb        creates the CSVs; needs network access
  analysis_multi.ipynb   reads the CSVs and does the analysis
  data/                  mcar_multi.csv, mar_multi.csv, mnar_multi.csv (committed)
  plots/                 PNGs written by the analysis (git-ignored)
```

`mechanism_variants/` is the exception. It holds the same four files but is not a
dataset: it varies the *generator* while holding the data fixed, so it sits alongside the
dataset folders rather than inside one of them.

The CSV files are committed, so the analysis notebooks run offline and reproduce exactly
what is shown. The generation notebooks only need to be re-run if the CSVs are deleted
or a generator parameter is changed.

## Datasets

The five datasets were chosen to stress different shapes rather than to repeat one
example. Titanic is small enough to read row by row, `breast_cancer` is wide and
continuous, `student_performance` is wide and categorical, `default_credit` is large and
anonymised, and `contraceptive_method` is small and categorical.

| Dataset | Source | Rows | Columns | Notes |
|---|---|---|---|---|
| `titanic` | seaborn | 712 | 5 | numeric only; the one dataset with both a univariate and a multivariate analysis |
| `breast_cancer` | scikit-learn | 569 | 30 | all continuous |
| `contraceptive_method` | UCI (id 30) | 1473 | 9 | categorical, factorized to integer codes |
| `student_performance` | UCI (id 320) | 649 | 30 | mixed types, factorized to integer codes |
| `default_credit` | UCI (id 350) | 30000 | 23 | anonymised `X1..X23` column names |

## Mechanisms

Missingness is injected with [`mdatagen`](https://github.com/ArthurMangussi/pymdatagen)
under a fixed seed of 42, reseeded immediately before each generator so that every
mechanism draws from the same starting point and is independent of what the others
consumed. Running a generation notebook top to bottom reproduces its committed CSVs
exactly. The parameters differ by dataset:

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

## The shared narrative

Every analysis notebook follows the same four sections, so the method is learned once
and applied five times:

1. **Exploratory Discovery** — `totals`, `bar`, `rate`, `upset` and `venn` confirm how
   much is missing and which columns go missing together.
2. **Mechanism Discovery** — `matrix`, `heatmap` in its three kinds, `dendrogram` and
   `parallel_coordinates` look for structure. Sorting the matrix by a candidate driver
   is the decisive step: a contiguous block means MAR, a diffuse spread does not.
3. **Comparative Discovery** — `density`, `boxplot` and `scatterplot` compare the
   observed distributions between the present and missing groups.
4. **Conclusions** — what the evidence supports for each of the three mechanisms.

`titanic/analysis_multi.ipynb` additionally closes with `littles_mcar_test()` and
`mann_whitney_test()`, and `student_performance/analysis_multi.ipynb` with
`littles_mcar_test()`, which put a p-value on conclusions the plots reach visually.

## Where each feature is demonstrated

Every multivariate analysis now exercises the whole package: all twelve plots, both
statistical tests, the metrics, and the non-default options. The table records where to
look, and — more usefully — where something is deliberately absent.

| Feature | Where |
|---|---|
| all 12 plots, both heatmap kinds beyond correlation | every multivariate analysis |
| `parallel_coordinates(kind="missingness")` | every multivariate analysis |
| `boxplot(kind="violin")`, `bar(measure="fraction")` | every analysis |
| `matrix(max_columns=)` | the four wide datasets |
| `littles_mcar_test()`, `mann_whitney_test()` | every analysis |
| `perfectly_correlated_missing_columns()`, `missing_pattern_counts()` | every multivariate analysis |
| metrics (`total_missing_rate`, `col_missing_rate`, `rows_complete`, …) | every analysis |

Two deliberate exceptions:

`titanic/analysis_uni.ipynb` has one incomplete column, so `upset`, `venn`,
`dendrogram`, the correlation and predictive heatmaps, `missing_pattern_counts()`,
`perfectly_correlated_missing_columns()` and the missingness mode of
`parallel_coordinates` have nothing to compare against. The notebook closes with a
section saying so, which makes it the one place in the set that shows when *not* to
reach for a plot.

`titanic/analysis_multi.ipynb` skips `max_columns`: with five columns there is nothing
to cap.

## What the statistical tests actually do here

Worth reading before trusting either test elsewhere. Across the five datasets,
`littles_mcar_test()` behaves badly more often than not:

| Dataset | Little's p on the MCAR data | Reading |
|---|---|---|
| `titanic` (both) | 0.62, 0.34 | correct, MCAR not rejected |
| `student_performance` | 0.24 | correct |
| `contraceptive_method` | 0.039 | **false positive** at the 0.05 threshold |
| `default_credit` | 2.1e-45 | **false positive**; 30000 rows gives it power to flag trivia |
| `breast_cancer` | 1.0 (and 1.0 for MAR) | **uninformative**; 30 columns over 569 rows degenerates the statistic |

`perfectly_correlated_missing_columns()` separates the mechanisms far more reliably: it
returns identical pairs under MAR in every dataset and none under MCAR or MNAR. The
distinct-pattern count does the same from the row side, collapsing from hundreds or
thousands under MCAR to a handful under MAR. Each notebook says this in its own
Conclusions.

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
* **The value-based views are what recover the difference.** `heatmap(kind="biserial")`
  and the distribution plots separate `uMAR.highest` from `uMAR.lowest` cleanly, where
  every mask-based plot shows an identical picture.
* **`mMNAR.median` defeats the whole toolkit on narrow data.** On `titanic` it is
  indistinguishable from `mMCAR.random` by every diagnostic available, Little's test
  included (p=0.482 against 0.618). It is caught on `breast_cancer`, but by a test that
  returns `nan` for the MCAR baseline on that same dataset.

Two `mdatagen` behaviours worth knowing are recorded there as well: `mMNAR.MBIR` overshoots
its requested rate (0.600 against 0.200 on `titanic`, leaving zero complete rows), and
`uMAR.rank()` raises `KeyError` unless the index is reset to `0..n-1`, which the gapped
index left by `dropna()` does not satisfy.

## Running them

Install the package together with the notebook dependencies:

```bash
pip install -e ".[notebooks]"
```

Then open any `analysis_*.ipynb` and run it top to bottom; the CSVs it needs are already
in `data/`. Run the matching `gen_*.ipynb` first only if those CSVs are missing, and
note that it downloads the source dataset.

The analysis notebooks write PNGs into `plots/`, which requires a browser for image
export. `plotly_get_chrome -y` fetches one if `kaleido` has not already done so.

`make notebook-check` executes `titanic/analysis_multi.ipynb` against the committed CSVs
without writing anything back. CI runs the same target, which is what keeps these
notebooks from drifting out of step with the package API.

`make docs` stages these notebooks into `docs/` and builds the site around them. The
staged copy under `docs/notebooks/` is generated and git-ignored; the originals here are
the source. Because the site renders the committed outputs rather than re-running the
notebooks, re-run them and commit the result whenever the analyses change, or the site
will keep showing the previous figures.
