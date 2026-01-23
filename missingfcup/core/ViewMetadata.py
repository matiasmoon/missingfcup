from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Literal
import pandas as pd
import numpy as np
import warnings

class OrderType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class NumericOrder(Enum):
    ASC = "asc"
    DESC = "desc"


@dataclass(frozen=True)
class OrderBySpec:
    column: str
    type: OrderType
    numeric_order: Optional[NumericOrder] = None
    category_order: Optional[List[str]] = None

@dataclass
class ViewMetadata:
    """
    Describes *what* data should be shown:
    - column selection
    - missingness / completeness filtering
    - column limits
    - ordering
    """

    # ------------------------------------------------------------------
    # Column selection & limits
    # ------------------------------------------------------------------
    selected_columns: Optional[List[str]] = None
    max_columns: int = 50

    # ------------------------------------------------------------------
    # Missingness / completeness logic
    # ------------------------------------------------------------------
    ignore_high_missingness: bool = True
    high_missingness_threshold: float = 0.9  # fraction missing

    completeness_mode: Optional[Literal["most", "least"]] = None
    completeness_threshold: float = 0.0
    max_columns_by_completeness: int = 0

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------
    order_by: Optional[List[OrderBySpec]] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.order_by is None:
            return

        if len(self.order_by) > 2:
            raise ValueError("order_by supports a maximum of 2 columns")

        for spec in self.order_by:
            if spec.type == OrderType.NUMERIC:
                if spec.numeric_order is None or spec.category_order is not None:
                    raise ValueError(
                        f"Numeric order requires numeric_order only (column={spec.column})"
                    )

            elif spec.type == OrderType.CATEGORICAL:
                if spec.category_order is None or spec.numeric_order is not None:
                    raise ValueError(
                        f"Categorical order requires category_order only (column={spec.column})"
                    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        n_rows = len(df)

        # ---- high missingness exclusion --------------------------------
        if self.ignore_high_missingness:
            missing_fraction = df.isna().mean()
            keep = missing_fraction < self.high_missingness_threshold

            if not keep.any():
                raise ValueError("All columns filtered by missingness threshold")

            df = df.loc[:, keep]

        # ---- explicit column selection ---------------------------------
        if self.selected_columns:
            missing = [c for c in self.selected_columns if c not in df.columns]
            if missing:
                warnings.warn(f"Skipping missing columns: {missing}")

            cols = [c for c in self.selected_columns if c in df.columns]
            if not cols:
                raise ValueError("No selected columns exist")

            df = df[cols]

        # ---- completeness-based filtering ------------------------------
        if self.completeness_mode:
            completeness = 1 - df.isna().mean()

            if self.completeness_mode == "most":
                if self.completeness_threshold > 0:
                    df = df.loc[:, completeness >= self.completeness_threshold]

                if self.max_columns_by_completeness > 0:
                    idx = np.argsort(completeness)[-self.max_columns_by_completeness :]
                    df = df.iloc[:, np.sort(idx)]

            elif self.completeness_mode == "least":
                if self.completeness_threshold > 0:
                    df = df.loc[:, completeness <= self.completeness_threshold]

                if self.max_columns_by_completeness > 0:
                    idx = np.argsort(completeness)[: self.max_columns_by_completeness]
                    df = df.iloc[:, np.sort(idx)]

        # ---- max columns hard limit ------------------------------------
        if df.shape[1] > self.max_columns:
            df = df.iloc[:, : self.max_columns]
            warnings.warn(f"Limiting to first {self.max_columns} columns")

        # ---- ordering --------------------------------------------------
        if self.order_by:
            for spec in reversed(self.order_by):
                if spec.type == OrderType.NUMERIC:
                    df = df.sort_values(
                        spec.column,
                        ascending=(spec.numeric_order == NumericOrder.ASC),
                        kind="stable",
                    )
                else:
                    cat = pd.Categorical(
                        df[spec.column],
                        categories=spec.category_order,
                        ordered=True,
                    )
                    df = df.assign(**{spec.column: cat}).sort_values(
                        spec.column, kind="stable"
                    )

        return df
