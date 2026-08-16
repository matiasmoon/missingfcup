"""A small built-in dataset, so the examples and a quick experiment do not need a
download or a file on disk.

The frame is written out as a literal rather than generated from a seeded RNG: the
values are then stable across numpy versions and can be read straight from the source.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["sample_data"]

_ROWS = [
    # age    income   score  visits  rating
    (22.0, None, 2.0, 3, 1.0),
    (25.0, None, None, 7, None),
    (31.0, 41000.0, 4.0, 1, 3.0),
    (None, 52000.0, 3.0, 4, 2.0),
    (44.0, 58000.0, None, 9, None),
    (28.0, None, 5.0, 2, 4.0),
    (39.0, 47000.0, 4.0, 6, 3.0),
    (51.0, 71000.0, 7.0, 5, 5.0),
    (None, 63000.0, 6.0, 8, 4.0),
    (35.0, 45000.0, None, 3, None),
    (47.0, 68000.0, 8.0, 2, 5.0),
    (29.0, None, 3.0, 7, 2.0),
    (56.0, 82000.0, 9.0, 4, 5.0),
    (None, 55000.0, 5.0, 6, 3.0),
    (33.0, 44000.0, 4.0, 1, 3.0),
    (61.0, 91000.0, 8.0, 9, 5.0),
    (26.0, None, None, 5, None),
    (48.0, 66000.0, 7.0, 2, 4.0),
    (None, 59000.0, 6.0, 8, 4.0),
    (42.0, 61000.0, 5.0, 3, 4.0),
]

_COLUMNS = ["age", "income", "score", "visits", "rating"]


def sample_data() -> pd.DataFrame:
    """Return a small DataFrame with missing values, for examples and experiments.

    20 rows and 5 numeric columns. Every plot in the package works on it. The gaps are
    placed deliberately so the plots have structure to show rather than noise:

    ==========  ==========================================================
    ``age``     missing in 4 scattered rows, unrelated to anything (MCAR)
    ``income``  missing exactly where ``age`` is young (MAR)
    ``score``   missing in its own low range (MNAR)
    ``visits``  never missing, a complete column
    ``rating``  missing together with ``score``
    ==========  ==========================================================

    A fresh copy is returned on each call, so callers can modify it freely.

    Examples
    --------
    >>> import missingfcup as mf
    >>> df = mf.sample_data()
    >>> df.shape
    (20, 5)
    >>> mf.matrix(df).show()   # doctest: +SKIP
    """
    return pd.DataFrame(_ROWS, columns=_COLUMNS)
