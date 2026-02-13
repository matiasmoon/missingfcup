import pandas as pd

from missingfcup.core.MissingData import MissingData


def main():
    print("\n=== MissingFCUP Heatmap Demo (10 Examples) ===")

    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )

    md = MissingData(df)

    # ------------------------------------------------------------------
    # 1️⃣ Very easy (minimal)
    # ------------------------------------------------------------------
    md.heatmap().show()

    # ------------------------------------------------------------------
    # 2️⃣-5️⃣ Simple with personalization
    # ------------------------------------------------------------------
    md.heatmap(
        selected_columns=["ZIP CODE", "BOROUGH", "OFF STREET NAME"],
        title="ZIP/Borough/Off-Street",
    ).show()

    md.heatmap(
        selected_columns=[
            "ON STREET NAME",
            "CROSS STREET NAME",
            "OFF STREET NAME",
        ],
        title="Street Fields",
        xgap=1,
        ygap=1,
    ).show()

    md.heatmap(
        selected_columns=[
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
            "CONTRIBUTING FACTOR VEHICLE 3",
        ],
        title="Contributing Factors (1-3)",
        show_colorscale=True,
    ).show()

    md.heatmap(
        selected_columns=[
            "VEHICLE TYPE CODE 1",
            "VEHICLE TYPE CODE 2",
            "VEHICLE TYPE CODE 3",
        ],
        title="Vehicle Types (1-3)",
        group_by_mode="missing",
    ).show()

    # ------------------------------------------------------------------
    # 6️⃣-🔟 More complex
    # ------------------------------------------------------------------
    md.heatmap(
        title="Columns Ordered by Missingness",
        order_by=[{"axis": "columns", "column": "__missing__", "ascending": False}],
    ).show()

    md.heatmap(
        title="Order by Numeric Column (DESC)",
        selected_columns=[
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
            "ZIP CODE",
            "BOROUGH",
        ],
        order_by=[{"axis": "rows", "column": "NUMBER OF PERSONS INJURED", "type": "numeric", "ascending": False}],
    ).show()

    md.heatmap(
        title="Order by Numeric Column (ASC)",
        selected_columns=[
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
            "ZIP CODE",
            "BOROUGH",
        ],
        order_by=[{"axis": "rows", "column": "NUMBER OF PERSONS INJURED", "type": "numeric", "ascending": True}],
    ).show()

    md.heatmap(
        title="Order by Categorical (Alphabetical)",
        selected_columns=[
            "BOROUGH",
            "ZIP CODE",
            "ON STREET NAME",
        ],
        order_by=[{"axis": "rows", "column": "BOROUGH", "type": "categorical", "ascending": True}],
    ).show()

    md.heatmap(
        title="Order by Categorical (Custom Order)",
        selected_columns=[
            "BOROUGH",
            "ZIP CODE",
            "ON STREET NAME",
        ],
        order_by=[{
            "axis": "rows",
            "column": "BOROUGH",
            "type": "categorical",
            "category_order": ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"],
        }],
    ).show()

    md.heatmap(
        title="Order by Two Columns (Numeric + Categorical)",
        selected_columns=[
            "BOROUGH",
            "ZIP CODE",
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
        ],
        order_by=[
            {"axis": "rows", "column": "BOROUGH", "type": "categorical", "ascending": True},
            {"axis": "rows", "column": "NUMBER OF PERSONS INJURED", "type": "numeric", "ascending": False},
        ],
    ).show()

    md.heatmap(
        title="Rows Ordered by Missingness",
        order_by=[{"axis": "rows", "column": "__missing__", "ascending": False}],
    ).show()

    md.heatmap(
        title="Rows + Columns Ordered by Missingness",
        order_by=[
            {"axis": "columns", "column": "__missing__", "ascending": False},
            {"axis": "rows", "column": "__missing__", "ascending": False},
        ],
    ).show()

    md.heatmap(
        title="Top Columns, Random Rows",
        order_by=[{"axis": "columns", "column": "__missing__", "ascending": False}],
        max_columns=12,
        show_colorscale=True,
        xgap=1,
        ygap=1,
    ).show()

    md.heatmap(
        selected_columns=[
            "ZIP CODE",
            "BOROUGH",
            "ON STREET NAME",
            "CROSS STREET NAME",
            "OFF STREET NAME",
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
        ],
        title="Location + Factors",
        background_color="#0b0b0b",
        text_color="#f2f2f2",
        show_colorscale=True,
    ).show()

if __name__ == "__main__":
    main()
