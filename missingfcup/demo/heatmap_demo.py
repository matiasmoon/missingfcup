import pandas as pd

from missingfcup.core.MissingData import MissingData
from missingfcup.plots.Panel import Panel


def main():
    print("\n=== MissingFCUP Heatmap Showcase ===")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )

    md = MissingData(df)

    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Overall missingness: {md.total_missingness:.2%}")

    # ------------------------------------------------------------------
    # Heatmap configurations
    # ------------------------------------------------------------------

    heatmap_basic = md.heatmap(
        title="All Columns (Binary Missingness)",
        max_rows=300,
    )

    heatmap_sorted_missing = md.heatmap(
        order_by=[{"column": "__missing__", "ascending": False}],
        title="Columns Ordered by Missingness",
        max_rows=300,
    )

    heatmap_filtered = md.heatmap(
        selected_columns=[
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
        ],
        title="Key Injury & Factor Columns",
        max_rows=300,
    )

    heatmap_missing_focus = md.heatmap(
        order_by=[{"column": "__missing__", "ascending": False}],
        group_by_mode="missing",
        show_colorscale=True,
        title="Missingness-Focused View",
        max_rows=300,
    )

    # ------------------------------------------------------------------
    # Panel view
    # ------------------------------------------------------------------
    panel = Panel(
        [
            heatmap_basic,
            heatmap_sorted_missing,
            heatmap_filtered,
            heatmap_missing_focus,
        ]
    )

    panel.show()
    # panel.save("heatmap_showcase.html")


if __name__ == "__main__":
    main()
