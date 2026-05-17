import pandas as pd
from missingfcup import MissingData


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv"
    )

    # Drop the row index column from Rdatasets
    if df.columns[0] in {"Unnamed: 0", "rownames"}:
        df = df.drop(columns=[df.columns[0]])

    data = MissingData(df)

    print("\n=== Parallel Coordinates Demo (airquality) ===")
    print("Missingness is mostly in Ozone and Solar.R.\n")

    cols = ["Ozone", "Solar.R", "Wind", "Temp", "Month", "Day"]

    # ------------------------------------------------------------------
    # 1️⃣ Minimal
    # ------------------------------------------------------------------
    data.parallel_coordinates(
        selected_columns=cols,
        missingness_color_column="Ozone",
    ).show()

    # ------------------------------------------------------------------
    # 1️⃣b Missingness-only view (0/1) with transparency
    # ------------------------------------------------------------------
    data.parallel_coordinates(
        selected_columns=cols,
        missingness_color_column="Ozone",
        missingness_only=True,
        title="Missing Data Patterns in Air Quality Dataset",
        line_opacity=0.2,
        width=1100,
        height=480,
    ).show()

    # ------------------------------------------------------------------
    # 2️⃣ Emphasize missingness in Solar.R
    # ------------------------------------------------------------------
    data.parallel_coordinates(
        selected_columns=cols,
        missingness_color_column="Solar.R",
        title="airquality: Missingness in Solar.R",
        present_color="#1b9e77",
        missing_color="#d95f02",
        line_opacity=0.6,
    ).show()

    # ------------------------------------------------------------------
    # 3️⃣ Compact layout
    # ------------------------------------------------------------------
    data.parallel_coordinates(
        selected_columns=cols,
        missingness_color_column="Ozone",
        title="airquality: Compact Layout",
        width=1000,
        height=450,
        line_opacity=0.55,
        show_colorbar=False,
    ).show()

    # ------------------------------------------------------------------
    # 4️⃣ Styling (background + text)
    # ------------------------------------------------------------------
    data.parallel_coordinates(
        selected_columns=cols,
        missingness_color_column="Ozone",
        title="airquality: Styled",
        background_color="#f7f3ee",
        text_color="#1f1f1f",
        present_color="#2c7fb8",
        missing_color="#d95f02",
        line_opacity=0.5,
    ).show()

    # ------------------------------------------------------------------
    # 5️⃣ Emulate paper-style view (10% below range)
    # ------------------------------------------------------------------
    data.parallel_coordinates(
        selected_columns=["Temp", "Ozone", "Solar.R", "Wind", "Month", "Day"],
        missingness_color_column="Ozone",
        title="airquality: Ozone Missingness (10% below range)",
        normalize=False,
        impute_below_range_frac=0.1,
        present_color="#1b9e77",
        missing_color="#d95f02",
        line_opacity=0.2,
        width=1150,
        height=520,
    ).show()

    # ------------------------------------------------------------------
    # 6️⃣ Focused subset (Temp/Wind/Ozone/Solar.R)
    # ------------------------------------------------------------------
    data.parallel_coordinates(
        selected_columns=["Temp", "Ozone", "Wind", "Solar.R"],
        missingness_color_column="Ozone",
        title="airquality: Ozone Missingness vs Temp/Wind/Solar.R",
        normalize=False,
        impute_below_range_frac=0.1,
        line_opacity=0.55,
        width=1000,
        height=450,
    ).show()

    # ------------------------------------------------------------------
    # 7️⃣ Alternate focus: Solar.R missingness
    # ------------------------------------------------------------------
    data.parallel_coordinates(
        selected_columns=["Temp", "Solar.R", "Ozone", "Wind", "Month", "Day"],
        missingness_color_column="Solar.R",
        title="airquality: Solar.R Missingness (10% below range)",
        normalize=False,
        impute_below_range_frac=0.1,
        present_color="#1b9e77",
        missing_color="#d95f02",
        line_opacity=0.5,
        width=1150,
        height=520,
    ).show()


if __name__ == "__main__":
    main()
