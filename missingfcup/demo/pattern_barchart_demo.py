# demo_pattern_barchart.py

import pandas as pd
import numpy as np

from missingfcup.core.MissingData import MissingData
from missingfcup.plots.PatternBarChart import PatternBarChart


def main():
    print("=== MissingFCUP Pattern Bar Chart Demo ===")

    rng = np.random.default_rng(0)

    df = pd.DataFrame({
        "A": rng.normal(size=300),
        "B": rng.normal(size=300),
        "C": rng.normal(size=300),
        "D": rng.normal(size=300),
    })

    # Inject structured missingness
    df.loc[rng.choice(df.index, 60, replace=False), "A"] = np.nan
    df.loc[rng.choice(df.index, 40, replace=False), "B"] = np.nan
    df.loc[rng.choice(df.index, 30, replace=False), "C"] = np.nan
    df.loc[rng.choice(df.index, 20, replace=False), ["A", "B"]] = np.nan

    data = MissingData(df)

    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print(f"Overall missingness: {data.total_missing_rate:.2%}")

    chart = PatternBarChart(
        data=data,
        title="Most common missingness patterns",
        max_patterns=10,
        background_color="#f0f0f0",
        text_color="#333333",
    )

    chart.show()


if __name__ == "__main__":
    main()