import uuid
import pandas as pd
from typing import Optional, List, Literal

from .matrix import Matrix
from ..plots.heatmap import Heatmap
from ..plots.barchart import BarChart

class MissingObject:
    """
    Main entry point for analyzing missing data patterns in your DataFrame.

    MissingObject wraps your DataFrame and provides convenient methods to:
    - Create missingness matrices
    - Generate visualizations (heatmaps, bar charts)
    - Analyze patterns of missing data

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> from missingfcup import MissingObject
    >>>
    >>> df = pd.read_csv("data.csv")
    >>> missing = MissingObject(df)
    >>>
    >>> # Show a heatmap
    >>> missing.heatmap().show()
    >>>
    >>> # Show a bar chart of completeness
    >>> missing.barchart().show()

    Customized analysis:

    >>> # Focus on 10 most incomplete columns
    >>> missing.heatmap(
    ...     completeness_mode="least",
    ...     max_columns_by_completeness=10,
    ...     figure_size_pixels=(1200, 800)
    ... ).show()
    """

    def __init__(self, data: pd.DataFrame, id=None):
        """
        Initialize a MissingObject for your DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to analyze for missing values
        id : optional
            Unique identifier for this analysis (auto-generated if not provided)
        """
        self.id = id or uuid.uuid4()
        self.data = data

    def matrix(
        self,
        max_columns: int = 50,
        selected_columns: Optional[List[str]] = None,
        completeness_mode: Optional[Literal["most", "least"]] = None,
        completeness_threshold: float = 0,
        max_columns_by_completeness: int = 0,
        ignore_high_missingness: bool = True,
        high_missingness_threshold: float = 0.9,
    ) -> Matrix:
        """
        Create a missingness matrix from the DataFrame.

        Parameters
        ----------
        max_columns : int, default=50
            Maximum number of columns to include.

        selected_columns : list of str, optional
            Specific columns to analyze by name.

        completeness_mode : {"most", "least"}, optional
            Focus on most complete or least complete columns.

        completeness_threshold : float, default=0
            Minimum/maximum completeness (0-1) required for columns.

        max_columns_by_completeness : int, default=0
            Limit to N most/least complete columns.

        ignore_high_missingness : bool, default=True
            Automatically exclude columns with extremely high missingness.

        high_missingness_threshold : float, default=0.9
            Missingness threshold for automatic exclusion (0 to 1).
            Default of 0.9 excludes columns with ≥90% missing values.

        Returns
        -------
        Matrix
            Binary matrix showing present (1) and missing (0) values.

        See Also
        --------
        Matrix.from_dataframe : Full documentation of all parameters
        """
        return Matrix.from_dataframe(
            self.data,
            max_columns=max_columns,
            selected_columns=selected_columns,
            completeness_mode=completeness_mode,
            completeness_threshold=completeness_threshold,
            max_columns_by_completeness=max_columns_by_completeness,
            ignore_high_missingness=ignore_high_missingness,
            high_missingness_threshold=high_missingness_threshold,
        )

    def heatmap(self, **kwargs) -> Heatmap:
        """
        Create an interactive missingness heatmap.

        This is a convenience method that creates both a Matrix and a Heatmap.

        Parameters
        ----------
        **kwargs
            Accepts all Matrix parameters (max_columns, selected_columns, etc.)
            and all Heatmap parameters (figure_size_pixels, colors, etc.)

        Returns
        -------
        Heatmap
            Interactive heatmap visualization. Call .show() to display.

        Examples
        --------
        Basic heatmap:
        >>> missing.heatmap().show()

        Customized heatmap:
        >>> missing.heatmap(
        ...     max_columns=20,
        ...     completeness_mode="most",
        ...     completeness_threshold=0.8,
        ...     figure_size_pixels=(1200, 800),
        ...     present_color="#4CAF50",
        ...     missing_color="#FF5722"
        ... ).show()

        Grouped heatmap to reveal patterns:
        >>> missing.heatmap(
        ...     group_by="age",
        ...     group_direction="ascending"
        ... ).show()

        See Also
        --------
        Heatmap : Full documentation of visualization parameters
        Matrix.from_dataframe : Full documentation of filtering parameters
        """
        # Extract Matrix parameters
        matrix_params = {
            "max_columns": kwargs.pop("max_columns", 50),
            "selected_columns": kwargs.pop("selected_columns", None),
            "completeness_mode": kwargs.pop("completeness_mode", None),
            "completeness_threshold": kwargs.pop("completeness_threshold", 0),
            "max_columns_by_completeness": kwargs.pop("max_columns_by_completeness", 0),
            "ignore_high_missingness": kwargs.pop("ignore_high_missingness", True),
            "high_missingness_threshold": kwargs.pop("high_missingness_threshold", 0.9),
        }

        # Create matrix
        matrix = self.matrix(**matrix_params)

        # Create and return heatmap with remaining kwargs (includes ordering params and group_by_mode)
        return Heatmap(matrix=matrix, **kwargs)

    def barchart(self, **kwargs) -> BarChart:
        """
        Create a bar chart showing completeness per column.

        Parameters
        ----------
        **kwargs
            BarChart configuration parameters

        Returns
        -------
        BarChart
            Bar chart visualization. Call .show() to display.

        Examples
        --------
        >>> missing.barchart().show()
        """
        return BarChart(self.data, **kwargs)
