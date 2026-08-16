"""Validation for the ``order_by`` option shared by :class:`_Matrix` and the bar charts.

The two plots used to read different keys from the same option: the matrix understood
``axis``/``ascending``, the bars understood ``numeric_order``, and neither complained
about a key it did not recognise. A spec written for one plot was therefore either
rejected by the other or, worse, silently ignored. Everything is normalised here
instead, and an unrecognised key is an error rather than a no-op.
"""

from __future__ import annotations

from typing import Dict, List, Optional

__all__ = ["normalize_order_by", "COLUMN_SENTINELS", "ROW_SENTINELS"]

_VALID_KEYS = {"column", "axis", "ascending", "direction", "type", "category_order"}
_VALID_AXES = {"columns", "rows"}
_VALID_TYPES = {"numeric", "categorical"}
_VALID_DIRECTIONS = {"asc": True, "desc": False}

COLUMN_SENTINELS = {"__missing__", "__column__"}
ROW_SENTINELS = {"__missing__", "__row__"}


def normalize_order_by(
    order_by: Optional[List[Dict]],
    *,
    supports_rows: bool = True,
    supports_data_columns: bool = True,
    max_specs: Optional[int] = None,
) -> List[Dict]:
    """Validate ``order_by`` and return it with every key filled in.

    Each returned spec has exactly ``column``, ``axis``, ``ascending``, ``type`` and
    ``category_order``, so callers can read them without ``.get`` defaults.

    Parameters
    ----------
    order_by : list of dict, optional
        The specs as the user wrote them. ``None`` or empty returns ``[]``.
    supports_rows : bool, default True
        Whether ``axis="rows"`` is meaningful for the calling plot.
    supports_data_columns : bool, default True
        Whether ordering by a named data column is meaningful. False for the bar
        charts, which aggregate per column, so row order cannot affect the output.
    max_specs : int, optional
        Upper bound on how many specs the plot can apply.

    Raises
    ------
    ValueError
        On an unknown key, a missing ``column``, a contradictory
        ``ascending``/``direction`` pair, or a value outside the allowed set.
    """
    if not order_by:
        return []

    if max_specs is not None and len(order_by) > max_specs:
        raise ValueError(f"order_by supports at most {max_specs} specifications.")

    normalized = []
    for position, spec in enumerate(order_by):
        if not isinstance(spec, dict):
            raise ValueError(f"order_by[{position}] must be a dict, got {type(spec).__name__}.")

        unknown = set(spec) - _VALID_KEYS
        if unknown:
            raise ValueError(
                f"order_by[{position}] has unknown key(s) {sorted(unknown)}. "
                f"Valid keys are {sorted(_VALID_KEYS)}."
            )

        if "column" not in spec:
            raise ValueError(f"order_by[{position}] must have a 'column'.")
        column = spec["column"]

        axis = spec.get("axis", "columns")
        if axis not in _VALID_AXES:
            raise ValueError(
                f"order_by[{position}] has axis={axis!r}; expected one of {sorted(_VALID_AXES)}."
            )
        if axis == "rows" and not supports_rows:
            raise ValueError(f"order_by[{position}]: this plot cannot order by rows.")

        ascending = _resolve_ascending(spec, position)

        spec_type = spec.get("type", "numeric")
        if spec_type not in _VALID_TYPES:
            raise ValueError(
                f"order_by[{position}] has type={spec_type!r}; "
                f"expected one of {sorted(_VALID_TYPES)}."
            )

        allowed_sentinels = COLUMN_SENTINELS | ROW_SENTINELS
        if not supports_data_columns and column not in allowed_sentinels:
            raise ValueError(
                f"order_by[{position}]: this plot can only order by "
                f"{sorted(allowed_sentinels)}, not by the data column {column!r}. "
                "It shows one bar per column, so the order of rows cannot change it."
            )

        normalized.append(
            {
                "column": column,
                "axis": axis,
                "ascending": ascending,
                "type": spec_type,
                "category_order": spec.get("category_order"),
            }
        )

    return normalized


def _resolve_ascending(spec: Dict, position: int) -> bool:
    """Read the sort direction from either ``ascending`` or the ``direction`` alias."""
    has_ascending = "ascending" in spec
    has_direction = "direction" in spec

    direction_value = None
    if has_direction:
        direction = spec["direction"]
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"order_by[{position}] has direction={direction!r}; "
                f"expected {sorted(_VALID_DIRECTIONS)}."
            )
        direction_value = _VALID_DIRECTIONS[direction]

    if has_ascending and has_direction and bool(spec["ascending"]) != direction_value:
        raise ValueError(
            f"order_by[{position}] sets ascending={spec['ascending']!r} and "
            f"direction={spec['direction']!r}, which contradict each other."
        )

    if has_ascending:
        return bool(spec["ascending"])
    if has_direction:
        return direction_value
    return True
