# missingfcup

`missingfcup` is a small utility to visualize missing data in pandas
DataFrames using interactive Plotly heatmaps.

This repository contains a lightweight implementation of a "missingness
matrix" that shows which cells are present vs missing.

## Install

TODO

## Usage

Basic usage example:

```python
import pandas as pd
from missingfcup import missing_matrix

df = pd.DataFrame({
		"A": [1, None, 3],
		"B": [None, 2, 3],
})

fig = missing_matrix(df)
fig.show()
```

See `demo_matrix.py` for a small runnable example and `missingfcup/matrix.py`
for the implementation and parameter options.

## Tests

Run the unit tests with `pytest` from the project root:

```bash
pytest -q
```

## Notes

- The project uses Plotly for interactive figures; ensure you have a suitable
	renderer configured (default opens in browser).
- Colors, sizing and column selection are configurable via function parameters.
