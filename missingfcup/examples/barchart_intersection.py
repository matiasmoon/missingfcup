import statsmodels.api as sm
import pandas as pd
from missingfcup import MissingData

def main():
    df = sm.datasets.get_rdataset("airquality").data

    # Drop the row index column from Rdatasets, if present
    if df.columns[0] in {"Unnamed: 0", "X"}:
        df = df.drop(columns=[df.columns[0]])

    data = MissingData(df)

    print("\n=== Barchart Intersection Demo (airquality) ===")
    print("Missingness is mostly in Ozone and Solar.R.\n")

    # ------------------------------------------------------------------
    # 1️⃣ Minimal
    # ------------------------------------------------------------------
    data.barchart_intersection().show()


    # ------------------------------------------------------------------
    # 2️⃣ Focused on known missing columns
    # ------------------------------------------------------------------
    data.barchart_intersection(
        selected_columns=["Ozone", "Solar.R"],
        title="airquality: Ozone vs Solar.R",
    ).show()

    # ------------------------------------------------------------------
    # 3️⃣ Limit intersections and tweak dot size
    # ------------------------------------------------------------------
    data.barchart_intersection(
        selected_columns=["Ozone", "Solar.R"],
        max_intersections=3,
        matrix_dot_size=14,
        matrix_line_width=4,
        title="airquality: Top Intersections",
    ).show()

    # ------------------------------------------------------------------
    # 4️⃣ Styling (background + colors)
    # ------------------------------------------------------------------
    data.barchart_intersection(
        selected_columns=["Ozone", "Solar.R"],
        background_color="#f7f3ee",
        text_color="#1f1f1f",
        missing_color="#2c3e50",
        excluded_dot_color="#c9c9c9",
        title="airquality: Styled",
    ).show()

    # ------------------------------------------------------------------
    # 5️⃣ Compact layout
    # ------------------------------------------------------------------
    data.barchart_intersection(
        selected_columns=["Ozone", "Solar.R"],
        width=900,
        height=500,
        title="airquality: Compact",
    ).show()

    # ------------------------------------------------------------------
    # 6️⃣-7️⃣ NYC collision factors examples
    # ------------------------------------------------------------------
    nyc_url = "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    nyc_df = pd.read_csv(nyc_url)
    nyc = MissingData(nyc_df)

    nyc.barchart_intersection(
        selected_columns=["ZIP CODE", "BOROUGH", "OFF STREET NAME"],
        title="NYC: ZIP/Borough/Off-Street",
        max_intersections=6,
    ).show()

    nyc.barchart_intersection(
        selected_columns=[
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
            "CONTRIBUTING FACTOR VEHICLE 3",
        ],
        title="NYC: Contributing Factors (1-3)",
        max_intersections=6,
    ).show()


if __name__ == "__main__":
    main()
