import uuid
import pandas as pd
from typing import Optional, List

from .matrix import Matrix
from ..plots.heatmap import Heatmap
from ..plots.barchart import BarChart

class MissingObject:
    """Core domain object representing missingness for a dataset."""

    def __init__(self, data: pd.DataFrame, id=None):
        self.id = id or uuid.uuid4()
        self.data = data

    def matrix(
        self,
        max_cols: int = 50,
        selected_columns: Optional[List[str]] = None,
    ) -> Matrix:
        return Matrix.from_dataframe(
            self.data,
            max_cols=max_cols,
            selected_columns=selected_columns,
        )

    def heatmap(self, **kwargs) -> Heatmap:
        matrix = self.matrix(
            max_cols=kwargs.pop("max_cols", 50),
            selected_columns=kwargs.pop("selected_columns", None),
        )
        return Heatmap(matrix=matrix, **kwargs)

    def barchart(self, **kwargs) -> BarChart:
        return BarChart(self.data, **kwargs)
