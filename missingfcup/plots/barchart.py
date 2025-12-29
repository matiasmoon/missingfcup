import pandas as pd
from .plot import Plot


class BarChart(Plot):
    """Placeholder for bar chart missingness summary."""

    def __init__(self, data: pd.DataFrame, **kwargs):
        self.data = data

    def show(self):
        print("Showing bar chart (not implemented yet)")

    def save(self, path: str):
        print(f"Saving bar chart to {path} (not implemented yet)")
