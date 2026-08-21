"""One hover grammar for every plot.

A tooltip is read mid-gesture, so it has to answer two questions in a fixed order:
**what am I pointing at**, then **what is it worth**. Every plot in the package
follows that shape, and the words it uses are the words already on the figure --
``NA`` and ``!NA``, never "Missing" and "Present", so a point never gets called one
thing by the legend and another by its tooltip.

The helpers here exist so the formatting lives in one place. A count is always shown
against its total, because "4" is unreadable without knowing whether the dataset has
twenty rows or twenty thousand, and rates always carry two decimals, matching the
values drawn inside the plots.
"""

from typing import List, Sequence

# Plotly ends a hover box with a right-hand "trace name" chip unless the template
# suppresses it. The trace name is already in the legend, so it only adds noise.
NO_TRACE_CHIP = "<extra></extra>"


def title(text: str) -> str:
    """The first line: what the cursor is on, emphasised."""
    return f"<b>{text}</b>"


def rows_of_total(count: float, total: int) -> str:
    """A count with the share it represents.

    Both, always. A bare count cannot be judged without the total, and a bare
    percentage hides how much data is behind it -- 50% of four rows and 50% of four
    thousand are not the same finding.
    """
    share = (count / total * 100) if total else 0.0
    return f"{count:,.0f} of {total:,} rows ({share:.2f}%)"


def rate(value: float, percentage: bool) -> str:
    """A rate on whichever scale the plot is drawing, with the package's precision."""
    return f"{value:.2f}%" if percentage else f"{value:.2f}"


def build(*lines: str) -> str:
    """Join the lines of a tooltip and suppress plotly's trace-name chip."""
    return "<br>".join(lines) + NO_TRACE_CHIP


def column_state(column: str, is_missing: bool) -> str:
    """``NA``/``!NA`` for one column, as the legends spell it."""
    return f"NA-{column}" if is_missing else f"!NA-{column}"


def customdata(*columns: Sequence) -> List[list]:
    """Zip per-point values into the row-wise shape plotly expects for customdata."""
    return [list(row) for row in zip(*columns)]
