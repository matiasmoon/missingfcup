import pandas as pd

from missingfcup.core.mixins._core import _MissingDataCoreMixin
from missingfcup.core.mixins._columns import _MissingDataColumnsMixin
from missingfcup.core.mixins._rows import _MissingDataRowsMixin
from missingfcup.core.mixins._utils import _MissingDataUtilsMixin
from missingfcup.core.mixins._plots import _MissingDataPlotMixin


class MissingData(
    _MissingDataCoreMixin,
    _MissingDataColumnsMixin,
    _MissingDataRowsMixin,
    _MissingDataUtilsMixin,
    _MissingDataPlotMixin,
):
    """
    Central class for analyzing and visualizing missing values in a pandas DataFrame.

    Provides cached metrics at the column, row, and dataset levels, along with
    pattern analysis, statistical tests, and plot factory methods.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze. Must be non-empty.

    Categorical variables
    ---------------------
    Plots that operate on the **missingness structure** (i.e. which cells are NaN)
    work transparently with any dtype, including categorical, object, and numeric:

        heatmap, barchart_count, barchart_rate, heatmap_rate,
        barchart_intersection, barchart_venn, dendrogram,
        heatmap_correlation, heatmap_predictive

    Plots that render **actual data values** require numeric columns:

        scatterplot, parallel_coordinates, boxplot, density, heatmap_biserial

    Encode categorical columns before passing to these plots, e.g.::

        df["col"] = pd.factorize(df["col"])[0]   # ordinal / nominal

    ``parallel_coordinates(missingness_only=True)`` is an escape hatch that
    renders all columns as binary (present/missing) regardless of dtype.

    Examples
    --------
    >>> import pandas as pd
    >>> from missingfcup import MissingData
    >>> df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})
    >>> md = MissingData(df)
    >>> md.col_missing_rate  # fraction missing per column
    a    0.333333
    b    0.333333
    dtype: float64
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame is empty")
        self.data = df
