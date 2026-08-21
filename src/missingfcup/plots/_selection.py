"""Shared column selection for the plots that show a chosen subset of columns.

Several plots (matrix, rate, dendrogram, and the association heatmaps) all pick which
columns to draw using the same options: drop the near-empty ones, keep an explicit
selection, drop columns with no variation in missingness, order by missing rate, and cap
the count. That logic used to live copied in each plot. It now lives here once.
"""

from typing import Iterable, List, Literal, Optional


def unusable_columns_error(param: str, requested: Iterable, available: Iterable, known: Iterable):
    """The error raised when none of the columns a caller named can be drawn.

    Names absent from the DataFrame and names removed by an earlier filter need
    different fixes, so they are reported separately rather than as one message that
    says neither.
    """
    known, available = set(known), set(available)
    absent = [c for c in requested if c not in known]
    filtered = [c for c in requested if c in known and c not in available]
    reasons = []
    if absent:
        reasons.append(f"not in the DataFrame: {absent}")
    if filtered:
        reasons.append(f"removed by an earlier filter: {filtered}")
    return ValueError(f"No usable {param} ({'; '.join(reasons)}).")


def select_columns(
    data,
    selected_columns: Optional[List[str]] = None,
    *,
    high_missingness_threshold: Optional[float] = None,
    drop_constant: bool = False,
    sort_by: Optional[Literal["missingness", "alphabetical"]] = None,
    ascending: bool = False,
    max_columns: int = 0,
    columns: Optional[List[str]] = None,
) -> List[str]:
    """Return the ordered list of column names a plot should display.

    The steps run in a fixed order: drop columns at or above
    ``high_missingness_threshold`` (``None`` skips that step),
    keep ``selected_columns``, drop columns whose missingness is constant, sort, then
    cap at ``max_columns`` (0 means no cap). ``columns`` sets the starting set and
    defaults to every column in ``data``.

    ``sort_by`` follows pandas: it names what to sort on and ``ascending`` gives the
    direction. ``None`` keeps the DataFrame's own column order.

    Raises ValueError if ``selected_columns`` is given but nothing survives the earlier
    filters, matching the message the individual plots used before.
    """
    miss_rate = data.col_missing_rate
    cols = list(columns) if columns is not None else list(data.columns)

    if high_missingness_threshold is not None:
        cols = [c for c in cols if miss_rate.get(c, 0.0) < high_missingness_threshold]

    if selected_columns is not None:
        chosen = [c for c in selected_columns if c in cols]
        if not chosen:
            raise unusable_columns_error("selected_columns", selected_columns, cols, data.columns)
        cols = chosen

    if drop_constant:
        mask = data.mask_missing
        cols = [c for c in cols if mask[c].nunique() > 1]

    if sort_by == "missingness":
        keep = set(cols)
        ordered = miss_rate[miss_rate.index.isin(keep)].sort_values(ascending=ascending)
        cols = list(ordered.index)
    elif sort_by == "alphabetical":
        cols = sorted(cols, reverse=not ascending)
    elif sort_by is not None:
        raise ValueError(f"sort_by must be 'missingness', 'alphabetical' or None, got {sort_by!r}.")

    if max_columns and max_columns > 0 and len(cols) > max_columns:
        cols = cols[:max_columns]

    return cols
