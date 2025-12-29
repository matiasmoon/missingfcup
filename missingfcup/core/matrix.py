import pandas as pd
from typing import Optional, List

class Matrix:
    """Represents a missingness matrix derived from a DataFrame."""

    def __init__(self, values, columns):
        self.values = values
        self.columns = columns

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        max_cols: int = 50,
        selected_columns: Optional[List[str]] = None,
    ) -> "Matrix":

        df = df.copy()

        if selected_columns:
            invalid = [c for c in selected_columns if c not in df.columns]
            if invalid:
                print(f"Warning: Columns {invalid} not found. Skipping them.")

            selected_columns = [c for c in selected_columns if c in df.columns]
            if selected_columns:
                df = df[selected_columns]

        if df.shape[1] > max_cols:
            df = df.iloc[:, :max_cols]

        boolean_matrix = df.notna().astype(int)

        return cls(
            values=boolean_matrix.values,
            columns=list(boolean_matrix.columns),
        )
