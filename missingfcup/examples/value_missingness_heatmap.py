import pandas as pd

from missingfcup import MissingData


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )
    md = MissingData(df)

    print("\n=== Value–Missingness Correlation Heatmap Demo (10 Examples) ===")
    print("Each cell [i, j] = point-biserial correlation between the observed")
    print("values of column i and the missingness indicator of column j.")
    print("Non-zero cells indicate the distribution of i differs between rows")
    print("where j is present vs. missing — a key MAR signal.\n")
    print("Note: non-numeric columns appear as NaN and are dropped by default.\n")

    # ------------------------------------------------------------------
    # 1️⃣ Very easy (minimal)
    # ------------------------------------------------------------------
    md.heatmap_value_missingness_correlation().show()

    # ------------------------------------------------------------------
    # 2️⃣-5️⃣ Simple with personalization
    # ------------------------------------------------------------------
    md.heatmap_value_missingness_correlation(
        title="Value–Missingness Association (All Columns)",
        show_values=True,
    ).show()

    md.heatmap_value_missingness_correlation(
        title="Value–Missingness: Injury Counts vs. Location Missingness",
        selected_value_columns=[
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
            "NUMBER OF PEDESTRIANS INJURED",
            "NUMBER OF CYCLIST INJURED",
            "NUMBER OF MOTORIST INJURED",
        ],
        selected_missing_columns=[
            "ZIP CODE",
            "BOROUGH",
            "ON STREET NAME",
            "CROSS STREET NAME",
            "OFF STREET NAME",
        ],
    ).show()

    md.heatmap_value_missingness_correlation(
        title="Injury Counts vs. Vehicle/Factor Missingness",
        selected_value_columns=[
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
            "NUMBER OF PEDESTRIANS INJURED",
            "NUMBER OF CYCLIST INJURED",
            "NUMBER OF MOTORIST INJURED",
        ],
        selected_missing_columns=[
            "CONTRIBUTING FACTOR VEHICLE 2",
            "CONTRIBUTING FACTOR VEHICLE 3",
            "VEHICLE TYPE CODE 2",
            "VEHICLE TYPE CODE 3",
        ],
        colorscale="RdBu",
    ).show()

    md.heatmap_value_missingness_correlation(
        title="Value–Missingness (No Values Shown)",
        show_values=False,
        order="asc",
    ).show()

    # ------------------------------------------------------------------
    # 6️⃣-🔟 More complex
    # ------------------------------------------------------------------
    md.heatmap_value_missingness_correlation(
        title="Top 10 Columns — Value vs. Missingness",
        max_columns=10,
        order="desc",
        show_values=True,
        value_round=2,
    ).show()

    md.heatmap_value_missingness_correlation(
        title="Injury Counts vs. All Missing Columns",
        selected_value_columns=[
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
            "NUMBER OF PEDESTRIANS INJURED",
            "NUMBER OF PEDESTRIANS KILLED",
            "NUMBER OF CYCLIST INJURED",
            "NUMBER OF CYCLIST KILLED",
            "NUMBER OF MOTORIST INJURED",
            "NUMBER OF MOTORIST KILLED",
        ],
        order_by_missingness=True,
        order="desc",
        value_round=2,
        show_colorbar=True,
    ).show()

    md.heatmap_value_missingness_correlation(
        title="Injury Counts vs. Location (No Colorbar)",
        selected_value_columns=[
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
        ],
        selected_missing_columns=[
            "ZIP CODE",
            "BOROUGH",
            "ON STREET NAME",
            "CROSS STREET NAME",
            "OFF STREET NAME",
        ],
        show_colorbar=False,
        show_values=True,
        value_round=2,
    ).show()

    md.heatmap_value_missingness_correlation(
        title="Value–Missingness (Custom NaN Color)",
        nan_color="#ffe0cc",
        colorscale="RdBu",
        show_values=True,
        max_columns=12,
    ).show()

    md.heatmap_value_missingness_correlation(
        title="Value–Missingness (Dark Theme)",
        background_color="#0b0b0b",
        text_color="#f2f2f2",
        colorscale="RdBu",
        nan_color="#2a2a2a",
        show_values=True,
        max_columns=12,
        width=950,
        height=600,
    ).show()


if __name__ == "__main__":
    main()
