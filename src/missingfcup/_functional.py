"""Flat functions over :class:`MissingData`, for a quick look at missing data.

Each function takes a DataFrame, builds a ``MissingData`` for you, and returns the
plot object. This matches how missingno is used (``msno.matrix(df)``)::

    import missingfcup as mf
    mf.matrix(df)
    mf.heatmap(df, kind="predictive")
    mf.bar(df, measure="rate")

If you work on the same DataFrame more than once, use the ``MissingData`` object
directly. It caches the masks and metrics, exposes the statistical tests, and composes
into a ``Panel``.

The functions here are kept in step with the plot methods on ``MissingData`` by a guard
test (see ``tests/test_smoke.py``).
"""

from __future__ import annotations

import pandas as pd

from missingfcup.core.missing_data import MissingData

__all__ = [
    "matrix",
    "heatmap",
    "rate",
    "bar",
    "totals",
    "upset",
    "venn",
    "dendrogram",
    "scatterplot",
    "density",
    "boxplot",
    "parallel_coordinates",
]


def matrix(df: pd.DataFrame, **kwargs):
    """Binary missingness matrix (nullity matrix). See ``MissingData.matrix``."""
    return MissingData(df).matrix(**kwargs)


def heatmap(df: pd.DataFrame, **kwargs):
    """Missingness association heatmap. See ``MissingData.heatmap`` (``kind=``)."""
    return MissingData(df).heatmap(**kwargs)


def rate(df: pd.DataFrame, **kwargs):
    """Missing rate per column as a single-row strip. See ``MissingData.rate``."""
    return MissingData(df).rate(**kwargs)


def bar(df: pd.DataFrame, **kwargs):
    """Per-column missingness bar chart. See ``MissingData.bar`` (``measure=``)."""
    return MissingData(df).bar(**kwargs)


def totals(df: pd.DataFrame, **kwargs):
    """Whole-dataset present/missing totals. See ``MissingData.totals``."""
    return MissingData(df).totals(**kwargs)


def upset(df: pd.DataFrame, **kwargs):
    """UpSet plot of missingness intersections. See ``MissingData.upset``."""
    return MissingData(df).upset(**kwargs)


def venn(df: pd.DataFrame, **kwargs):
    """Exclusive 3-column missingness subsets. See ``MissingData.venn``."""
    return MissingData(df).venn(**kwargs)


def dendrogram(df: pd.DataFrame, **kwargs):
    """Dendrogram of missingness correlation. See ``MissingData.dendrogram``."""
    return MissingData(df).dendrogram(**kwargs)


def scatterplot(df: pd.DataFrame, x: str, y: str, **kwargs):
    """Scatter plot keeping missing values visible. See ``MissingData.scatterplot``."""
    return MissingData(df).scatterplot(x, y, **kwargs)


def density(df: pd.DataFrame, x: str, color_by: str, **kwargs):
    """KDE density split by missingness. See ``MissingData.density``."""
    return MissingData(df).density(x, color_by, **kwargs)


def boxplot(df: pd.DataFrame, x: str, color_by: str, **kwargs):
    """Box/violin split by missingness. See ``MissingData.boxplot``."""
    return MissingData(df).boxplot(x, color_by, **kwargs)


def parallel_coordinates(df: pd.DataFrame, **kwargs):
    """Parallel coordinates colored by missingness. See ``MissingData.parallel_coordinates``."""
    return MissingData(df).parallel_coordinates(**kwargs)
