import pandas as pd

from missingfcup import MissingData


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )
    md = MissingData(df)

    print("\n=== Missingness Dendrogram Demo (6 Examples) ===")

    # ------------------------------------------------------------------
    # 1️⃣ Very easy (minimal)
    # ------------------------------------------------------------------
    md.dendrogram_missingness().show()

    # ------------------------------------------------------------------
    # 2️⃣-3️⃣ Simple with personalization
    # ------------------------------------------------------------------
    md.dendrogram_missingness(
        selected_columns=["ZIP CODE", "BOROUGH", "OFF STREET NAME", "ON STREET NAME"],
        title="Location Fields",
    ).show()

    md.dendrogram_missingness(
        selected_columns=[
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
            "CONTRIBUTING FACTOR VEHICLE 3",
            "CONTRIBUTING FACTOR VEHICLE 4",
        ],
        title="Contributing Factors (1-4)",
        linkage_method="complete",
    ).show()

    # ------------------------------------------------------------------
    # 4️⃣-6️⃣ More complex
    # ------------------------------------------------------------------
    md.dendrogram_missingness(
        max_columns=15,
        title="Top 15 Columns by Missingness",
    ).show()

    md.dendrogram_missingness(
        max_columns=15,
        title="Absolute Correlation (|corr|)",
        use_abs_correlation=True,
        linkage_method="average",
    ).show()

    md.dendrogram_missingness(
        selected_columns=[
            "ZIP CODE",
            "BOROUGH",
            "ON STREET NAME",
            "CROSS STREET NAME",
            "OFF STREET NAME",
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
            "VEHICLE TYPE CODE 1",
            "VEHICLE TYPE CODE 2",
        ],
        title="Location + Factors + Vehicle Types (Dark Theme)",
        background_color="#0b0b0b",
        text_color="#f2f2f2",
        line_color="#E45756",
    ).show()


if __name__ == "__main__":
    main()
