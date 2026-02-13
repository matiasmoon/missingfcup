import pandas as pd

from missingfcup.core.MissingData import MissingData


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )
    md = MissingData(df)

    print("\n=== Column Missing Rate Heatmap Demo (10 Examples) ===")

    # ------------------------------------------------------------------
    # 1️⃣ Very easy (minimal)
    # ------------------------------------------------------------------
    md.column_missing_rate_heatmap().show()

    # ------------------------------------------------------------------
    # 2️⃣-5️⃣ Simple with personalization
    # ------------------------------------------------------------------
    md.column_missing_rate_heatmap(
        selected_columns=["ZIP CODE", "BOROUGH", "OFF STREET NAME"],
        title="ZIP/Borough/Off-Street Missingness",
    ).show()

    md.column_missing_rate_heatmap(
        selected_columns=[
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
            "CONTRIBUTING FACTOR VEHICLE 3",
            "CONTRIBUTING FACTOR VEHICLE 4",
        ],
        title="Contributing Factors (1-4)",
        colorscale="Reds",
        show_values=True,
    ).show()

    md.column_missing_rate_heatmap(
        selected_columns=[
            "VEHICLE TYPE CODE 1",
            "VEHICLE TYPE CODE 2",
            "VEHICLE TYPE CODE 3",
            "VEHICLE TYPE CODE 4",
        ],
        title="Vehicle Types (1-4)",
        scale="percentage",
        value_round=1,
    ).show()

    md.column_missing_rate_heatmap(
        selected_columns=[
            "ON STREET NAME",
            "CROSS STREET NAME",
            "OFF STREET NAME",
            "ZIP CODE",
        ],
        title="Street Names + ZIP",
        colorscale="Viridis",
        show_colorbar=True,
    ).show()

    # ------------------------------------------------------------------
    # 6️⃣-🔟 More complex
    # ------------------------------------------------------------------
    md.column_missing_rate_heatmap(
        max_columns=12,
        title="Top 12 Columns by Missingness",
        show_values=False,
        order="desc",
    ).show()

    md.column_missing_rate_heatmap(
        max_columns=12,
        title="Top 12 (No Colorbar)",
        show_values=False,
        show_colorbar=False,
    ).show()

    md.column_missing_rate_heatmap(
        selected_columns=[
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
            "CONTRIBUTING FACTOR VEHICLE 3",
            "CONTRIBUTING FACTOR VEHICLE 4",
            "CONTRIBUTING FACTOR VEHICLE 5",
            "VEHICLE TYPE CODE 1",
            "VEHICLE TYPE CODE 2",
        ],
        title="Contributing Factors + Vehicle Types",
        max_columns=7,
        colorscale="Reds",
        value_round=2,
    ).show()

    md.column_missing_rate_heatmap(
        selected_columns=[
            "ZIP CODE",
            "BOROUGH",
            "ON STREET NAME",
            "CROSS STREET NAME",
            "OFF STREET NAME",
        ],
        title="Location Fields (Dark Theme)",
        background_color="#0b0b0b",
        text_color="#f2f2f2",
        colorscale="Reds",
    ).show()

    md.column_missing_rate_heatmap(
        max_columns=16,
        title="Broad Overview (16 Columns)",
        show_values=False,
    ).show()


if __name__ == "__main__":
    main()
