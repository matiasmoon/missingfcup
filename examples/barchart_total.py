import pandas as pd

from missingfcup import MissingData


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )
    md = MissingData(df)

    print("\n=== Overall Missingness Bar Chart Demo (5 Examples) ===")

    # ------------------------------------------------------------------
    # 1️⃣ Very easy (minimal)
    # ------------------------------------------------------------------
    md.barchart_total().show()

    # ------------------------------------------------------------------
    # 2️⃣-3️⃣ Simple with personalization
    # ------------------------------------------------------------------
    md.barchart_total(
        title="NYC Collisions — Overall Data Completeness",
    ).show()

    md.barchart_total(
        title="Overall Missingness (Custom Colors)",
        present_color="#4C78A8",
        missing_color="#F58518",
    ).show()

    # ------------------------------------------------------------------
    # 4️⃣-5️⃣ More complex
    # ------------------------------------------------------------------
    md.barchart_total(
        title="Overall Missingness (Wide)",
        width=700,
        height=450,
    ).show()

    md.barchart_total(
        title="Overall Missingness (Dark Theme)",
        background_color="#0b0b0b",
        text_color="#f2f2f2",
        present_color="#17becf",
        missing_color="#e377c2",
        width=600,
        height=420,
    ).show()


if __name__ == "__main__":
    main()
