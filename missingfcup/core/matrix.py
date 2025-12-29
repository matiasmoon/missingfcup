import pandas as pd
import numpy as np
from typing import Optional, List, Literal
import warnings


class Matrix:
    """
    Represents a missingness matrix derived from a DataFrame.

    A missingness matrix is a binary grid showing which values are present (1)
    or missing (0) in your data. Each row represents a record, each column
    represents a variable.

    Examples
    --------
    Basic usage - convert a DataFrame to a missingness matrix:

    >>> matrix = Matrix.from_dataframe(df)

    Limit to 20 columns maximum:

    >>> matrix = Matrix.from_dataframe(df, max_columns=20)

    Focus on the 10 most complete columns:

    >>> matrix = Matrix.from_dataframe(
    ...     df,
    ...     completeness_mode="most",
    ...     max_columns_by_completeness=10
    ... )

    Only show columns that are at least 80% complete:

    >>> matrix = Matrix.from_dataframe(
    ...     df,
    ...     completeness_mode="most",
    ...     completeness_threshold=0.8
    ... )

    Focus on specific columns by name:

    >>> matrix = Matrix.from_dataframe(
    ...     df,
    ...     selected_columns=["age", "income", "education"]
    ... )
    """

    def __init__(self, values, columns):
        """
        Initialize a Matrix object.

        Parameters
        ----------
        values : numpy.ndarray
            Binary matrix (0 = missing, 1 = present)
        columns : list
            Column names corresponding to the matrix columns
        """
        self.values = values
        self.columns = columns

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        max_columns: int = 50,
        selected_columns: Optional[List[str]] = None,
        completeness_mode: Optional[Literal["most", "least"]] = None,
        completeness_threshold: float = 0,
        max_columns_by_completeness: int = 0,
    ) -> "Matrix":
        """
        Create a missingness matrix from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to analyze for missing values.

        max_columns : int, default=50
            Maximum number of columns to include in the matrix.
            Prevents overwhelming visualizations with too many columns.
            If DataFrame has more columns, only the first `max_columns` are kept
            (after applying other filters).

        selected_columns : list of str, optional
            Specific column names to include in the matrix.
            If provided, only these columns will be analyzed.
            Columns not found in the DataFrame will be skipped with a warning.
            Example: ["age", "income", "city"]

        completeness_mode : {"most", "least"}, optional
            Which columns to prioritize based on completeness:
            - "most": Focus on columns with the most data (high completeness)
            - "least": Focus on columns with the least data (high missingness)
            - None: No filtering by completeness (default)

        completeness_threshold : float, default=0
            Minimum (or maximum) completeness required for columns to be included.
            Range: 0 to 1, where 1 = 100% complete, 0 = 100% missing.

            Behavior depends on `completeness_mode`:
            - If mode="most": Keep columns with completeness >= threshold
              Example: threshold=0.8 keeps columns that are at least 80% complete
            - If mode="least": Keep columns with completeness <= threshold
              Example: threshold=0.2 keeps columns that are at most 20% complete

            Ignored if `completeness_mode` is None.

        max_columns_by_completeness : int, default=0
            Limit the number of columns based on completeness ranking.
            Set to 0 to disable this filter (default).

            Behavior depends on `completeness_mode`:
            - If mode="most": Keep the N most complete columns
              Example: 10 keeps the 10 columns with highest completeness
            - If mode="least": Keep the N least complete columns
              Example: 10 keeps the 10 columns with most missingness

            Applied after `completeness_threshold` filter.
            Ignored if `completeness_mode` is None.

        Returns
        -------
        Matrix
            A Matrix object containing:
            - values: Binary numpy array (0 = missing, 1 = present)
            - columns: List of column names

        Raises
        ------
        TypeError
            If `df` is not a pandas DataFrame
        ValueError
            If DataFrame is empty, or if no columns meet the specified criteria

        Notes
        -----
        Filtering Pipeline:
        The function applies filters in this order:
        1. selected_columns: Choose specific columns (if provided)
        2. completeness_threshold: Filter by completeness percentage
        3. max_columns_by_completeness: Limit to N most/least complete
        4. max_columns: Hard cap on total number of columns

        Nullity Detection:
        Uses pandas .notna() to detect presence. All pandas null types
        (NaN, None, NaT, pd.NA) are treated as missing (0 in the matrix).

        Examples
        --------
        >>> # Simple: just create a matrix
        >>> matrix = Matrix.from_dataframe(df)

        >>> # Show only 10 most complete columns
        >>> matrix = Matrix.from_dataframe(
        ...     df,
        ...     completeness_mode="most",
        ...     max_columns_by_completeness=10
        ... )

        >>> # Show columns that are at least 50% complete, limit to 20
        >>> matrix = Matrix.from_dataframe(
        ...     df,
        ...     completeness_mode="most",
        ...     completeness_threshold=0.5,
        ...     max_columns=20
        ... )

        >>> # Focus on problematic columns (less than 30% complete)
        >>> matrix = Matrix.from_dataframe(
        ...     df,
        ...     completeness_mode="least",
        ...     completeness_threshold=0.3
        ... )
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

        if df.empty:
            raise ValueError("DataFrame is empty")

        if max_columns < 1:
            raise ValueError(f"max_columns must be at least 1, got {max_columns}")

        if completeness_mode is not None and completeness_mode not in ["most", "least"]:
            raise ValueError(f"completeness_mode must be 'most', 'least', or None, got '{completeness_mode}'")

        if completeness_threshold < 0 or completeness_threshold > 1:
            raise ValueError(f"completeness_threshold must be in range [0, 1], got {completeness_threshold}")

        if max_columns_by_completeness < 0:
            raise ValueError(f"max_columns_by_completeness must be non-negative, got {max_columns_by_completeness}")
    
        df = df.copy()

        # Handle selected columns
        if selected_columns:
            invalid = [c for c in selected_columns if c not in df.columns]
            if invalid:
                warnings.warn(
                    f"Columns {invalid} not found. Skipping them.",
                    UserWarning
                )

            selected_columns = [c for c in selected_columns if c in df.columns]
            if not selected_columns:
                raise ValueError("None of the selected columns exist in DataFrame")

            df = df[selected_columns]

        # Apply completeness filter if specified
        if completeness_mode is not None:
            if completeness_mode == 'most':
                # Filter by threshold first
                if completeness_threshold:
                    completeness = df.count(axis='rows').values / len(df)
                    df = df.iloc[:, completeness >= completeness_threshold]
                    if df.shape[1] == 0:
                        raise ValueError(f"No columns meet completeness threshold of {completeness_threshold}")

                # Then limit to N most complete columns
                if max_columns_by_completeness:
                    n_cols = min(max_columns_by_completeness, df.shape[1])
                    top_indices = np.argsort(df.count(axis='rows').values)[-n_cols:]
                    df = df.iloc[:, np.sort(top_indices)]

            elif completeness_mode == 'least':
                # Filter by threshold first
                if completeness_threshold:
                    completeness = df.count(axis='rows').values / len(df)
                    df = df.iloc[:, completeness <= completeness_threshold]
                    if df.shape[1] == 0:
                        raise ValueError(f"No columns meet incompleteness threshold of {completeness_threshold}")

                # Then limit to N least complete columns
                if max_columns_by_completeness:
                    n_cols = min(max_columns_by_completeness, df.shape[1])
                    bottom_indices = np.argsort(df.count(axis='rows').values)[:n_cols]
                    df = df.iloc[:, np.sort(bottom_indices)]

        # Apply max_columns limit (hard cap)
        if df.shape[1] > max_columns:
            df = df.iloc[:, :max_columns]
            warnings.warn(
                f"DataFrame has more than {max_columns} columns. "
                f"Only showing the first {max_columns}. "
                f"Increase 'max_columns' parameter to see more.",
                UserWarning
            )

        boolean_matrix = df.notna().astype(int)

        return cls(
            values=boolean_matrix.values,
            columns=list(boolean_matrix.columns),
        )
