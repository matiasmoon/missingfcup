import pandas as pd
from missingfcup import MissingData


def main():
    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )

    md = MissingData(df)

    print("\n=== Missing Rate by Column (Bar Chart) ===")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Overall missingness: {md.total_missing_rate:.2%}")

    # ------------------------------------------------------------------
    # 1️⃣ Default (fraction scale)
    # ------------------------------------------------------------------
    md.barchart_rate().show()

    # ------------------------------------------------------------------
    # 2️⃣ Percentage scale
    # ------------------------------------------------------------------
    md.barchart_rate(
        scale="percentage",
        title="Missing Rate by Column (%)",
    ).show()

    # ------------------------------------------------------------------
    # 3️⃣ Fraction scale, most missing first
    # ------------------------------------------------------------------
    md.barchart_rate(
        order_by=[{"column": "__missing__", "type": "numeric", "ascending": False}],
        max_columns=15,
        title="Missing Rate (Top 15, Most Missing First)",
    ).show()

    # ------------------------------------------------------------------
    # 4️⃣ Percentage scale, selected columns
    # ------------------------------------------------------------------
    md.barchart_rate(
        selected_columns=[
            "ZIP CODE",
            "BOROUGH",
            "ON STREET NAME",
            "CROSS STREET NAME",
            "OFF STREET NAME",
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
        ],
        scale="percentage",
        title="Missing Rate — Location & Factor Columns (%)",
    ).show()

    # ------------------------------------------------------------------
    # 5️⃣ Fraction scale, horizontal orientation, dark theme
    # ------------------------------------------------------------------
    md.barchart_rate(
        order_by=[{"column": "__missing__", "type": "numeric", "ascending": False}],
        max_columns=12,
        orientation="horizontal",
        scale="fraction",
        title="Missing Rate (Horizontal, Dark Theme)",
        background_color="#0b0b0b",
        text_color="#f2f2f2",
        missing_color="#e377c2",
        width=800,
        height=500,
    ).show()


if __name__ == "__main__":
    main()
