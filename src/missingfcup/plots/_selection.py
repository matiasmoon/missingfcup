"""Shared column selection for the plots that show a chosen subset of columns.

Several plots (matrix, rate, dendrogram, and the association heatmaps) all pick which
columns to draw using the same options: drop the near-empty ones, keep an explicit
selection, drop columns with no variation in missingness, order by missing rate, and cap
the count. That logic used to live copied in each plot. It now lives here once.
"""

from typing import List, Literal, Optional


def select_columns(
    data,
    selected_columns: Optional[List[str]] = None,
    *,
    ignore_high_missingness: bool = False,
    high_missingness_threshold: float = 0.9,
    drop_constant: bool = False,
    order_by_missingness: bool = False,
    order: Literal["desc", "asc"] = "desc",
    max_columns: int = 0,
    columns: Optional[List[str]] = None,
) -> List[str]:
    """Return the ordered list of column names a plot should display.

    The steps run in a fixed order: drop columns above ``high_missingness_threshold``,
    keep ``selected_columns``, drop columns whose missingness is constant, order by
    missing rate, then cap at ``max_columns`` (0 means no cap). ``columns`` sets the
    starting set and defaults to every column in ``data``.

    Raises ValueError if ``selected_columns`` is given but nothing survives the earlier
    filters, matching the message the individual plots used before.
    """
    miss_rate = data.col_missing_rate
    cols = list(columns) if columns is not None else list(data.columns)

    if ignore_high_missingness:
        cols = [c for c in cols if miss_rate.get(c, 0.0) < high_missingness_threshold]

    if selected_columns is not None:
        chosen = [c for c in selected_columns if c in cols]
        if not chosen:
            raise ValueError("No selected_columns found in DataFrame.")
        cols = chosen

    if drop_constant:
        mask = data.mask_missing
        cols = [c for c in cols if mask[c].nunique() > 1]

    if order_by_missingness:
        keep = set(cols)
        ordered = miss_rate[miss_rate.index.isin(keep)].sort_values(ascending=order == "asc")
        cols = list(ordered.index)

    if max_columns and max_columns > 0 and len(cols) > max_columns:
        cols = cols[:max_columns]

    return cols
