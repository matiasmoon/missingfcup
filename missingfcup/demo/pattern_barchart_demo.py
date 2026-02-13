import pandas as pd

from missingfcup.core.MissingData import MissingData


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )

    data = MissingData(df)

    print("\n=== Pattern Bar Chart Demo (10 Examples) ===")
    print("Each plot shows 7 bars for 3 columns:")
    print("- 3 single-column missing patterns")
    print("- 3 two-column combinations")
    print("- 1 three-column combination\n")

    # ------------------------------------------------------------------
    # 1️⃣ Very easy (minimal)
    # ------------------------------------------------------------------
    data.pattern_barchart().show()

    # ------------------------------------------------------------------
    # 2️⃣-5️⃣ Simple with personalization
    # ------------------------------------------------------------------
    data.pattern_barchart(
        selected_columns=["ZIP CODE", "BOROUGH", "OFF STREET NAME"],
        title="ZIP/Borough/Off-Street Missingness",
    ).show()

    data.pattern_barchart(
        selected_columns=["ON STREET NAME", "CROSS STREET NAME", "OFF STREET NAME"],
        title="Street Fields Missingness",
        missing_color="#1f77b4",
    ).show()

    data.pattern_barchart(
        selected_columns=[
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
            "CONTRIBUTING FACTOR VEHICLE 3",
        ],
        title="Contributing Factors (1-3)",
        value="percent",
        show_values=True,
    ).show()

    data.pattern_barchart(
        selected_columns=["VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3"],
        title="Vehicle Type Codes (1-3)",
        max_label_length=32,
    ).show()

    # ------------------------------------------------------------------
    # 6️⃣-🔟 More complex
    # ------------------------------------------------------------------
    data.pattern_barchart(
        selected_columns=[
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 4",
            "CONTRIBUTING FACTOR VEHICLE 5",
        ],
        title="Contributing Factors (1,4,5)",
        value="percent",
        width=1100,
        height=520,
    ).show()

    data.pattern_barchart(
        selected_columns=[
            "VEHICLE TYPE CODE 1",
            "VEHICLE TYPE CODE 4",
            "VEHICLE TYPE CODE 5",
        ],
        title="Vehicle Type Codes (1,4,5)",
        sort_order="asc",
        background_color="#f8f5f0",
        text_color="#1b1b1b",
    ).show()

    data.pattern_barchart(
        selected_columns=["ZIP CODE", "ON STREET NAME", "CROSS STREET NAME"],
        title="ZIP + Street Names",
        width=900,
        height=520,
        missing_color="#d62728",
        show_values=True,
    ).show()

    data.pattern_barchart(
        selected_columns=["BOROUGH", "ZIP CODE", "ON STREET NAME"],
        title="Location Fields (Borough/ZIP/On-Street)",
        width=960,
        height=520,
        background_color="#0b0b0b",
        text_color="#f2f2f2",
        missing_color="#ff7f0e",
    ).show()

    data.pattern_barchart(
        selected_columns=["OFF STREET NAME", "CROSS STREET NAME", "VEHICLE TYPE CODE 2"],
        title="Off/Cross Street + Vehicle Type 2",
        width=950,
        height=520,
        show_legend=False,
        max_label_length=28,
    ).show()


if __name__ == "__main__":
    main()
