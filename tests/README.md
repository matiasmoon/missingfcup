# Tests

What each file covers, and why it exists. Run everything with `make test`.

| File | Covers | Why it exists |
|---|---|---|
| `test_plots.py` | every public plot | asserts what is *inside* each figure — trace values, series counts, axis config. A test that only checks `fig is not None` passes on a wrong figure. |
| `test_metrics.py` | the metric methods | checks the numbers against a 5x3 fixture computed by hand in the module docstring, so a wrong formula fails rather than a wrong type. |
| `test_shared.py` | column selection and label truncation | these helpers run underneath most plots, so a bug here surfaces as a dozen unrelated plot failures. |
| `test_examples.py` | `examples/*.py` | the examples are the documentation, so a change that breaks one fails the build instead of being found by the next reader. |
| `test_example_coverage.py` | example completeness | every public parameter must appear in a runnable example, since that is the one place a parameter is both described and proven. |

Each file opens with a docstring giving the fixture and the reasoning in full; this table
is the index, not the explanation.

## Running

```bash
make test               # the whole suite
pytest tests/test_plots.py -v
pytest tests/ -k heatmap
```

One test exports a PNG and therefore needs a browser. `kaleido` downloads one on first
use; `plotly_get_chrome -y` does it ahead of time if the download is inconvenient mid-run.

## Adding a plot

`test_plots.py` keeps a `PLOTS` table near the top listing every public plot and how to
build it. Adding an entry there gives the new plot the whole battery of shared checks —
non-empty traces, titles, sizing, label truncation — before any plot-specific test is
written. `test_example_coverage.py` will then require an example for each of its
parameters, which is the intended order: plot, example, then the specific assertions.
