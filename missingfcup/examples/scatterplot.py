import pandas as pd

from missingfcup import MissingData


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )

    data = MissingData(df)

    print("\n=== Scatter Plot Demo (10 Examples) ===")
    print("Dataset notes (observed missingness):")
    print("- NUMBER OF CYCLISTS INJURED/KILLED are fully missing (100%).")
    print("- ZIP CODE has ~5% missing.")
    print("- LATITUDE/LONGITUDE and most injury counts have no missing values.\n")

    # ------------------------------------------------------------------
    # 1️⃣ Very easy (minimal)
    # ------------------------------------------------------------------
    data.scatterplot(
        x="NUMBER OF PERSONS INJURED",
        y="NUMBER OF PERSONS KILLED",
    ).show()

    # ------------------------------------------------------------------
    # 2️⃣-5️⃣ Simple with personalization
    # ------------------------------------------------------------------
    data.scatterplot(
        x="ZIP CODE",
        y="LATITUDE",
        title="ZIP Code vs Latitude (Missing ZIP Only)",
    ).show()

    data.scatterplot(
        x="ZIP CODE",
        y="LONGITUDE",
        title="ZIP Code vs Longitude (Missing ZIP Only)",
        point_size=10,
    ).show()

    data.scatterplot(
        x="NUMBER OF PEDESTRIANS INJURED",
        y="NUMBER OF PEDESTRIANS KILLED",
        title="Pedestrians: Injured vs Killed (Complete)",
        present_color="#1f77b4",
        missing_color="#ff7f0e",
    ).show()

    data.scatterplot(
        x="NUMBER OF PERSONS INJURED",
        y="ZIP CODE",
        title="People Injured vs ZIP Code (Missing ZIP Only)",
        legend_title="Record Status",
    ).show()

    # ------------------------------------------------------------------
    # 6️⃣-🔟 More complex
    # ------------------------------------------------------------------
    data.scatterplot(
        x="LATITUDE",
        y="LONGITUDE",
        title="Latitude vs Longitude (Complete)",
        width=1100,
        height=500,
        point_size=9,
        axis_padding=0.08,
    ).show()

    data.scatterplot(
        x="NUMBER OF CYCLISTS INJURED",
        y="NUMBER OF CYCLISTS KILLED",
        title="Cyclists: Injured vs Killed (All Missing)",
        width=900,
        height=520,
        background_color="#f8f5f0",
        text_color="#1b1b1b",
        point_size=8,
        axis_padding=0.1,
    ).show()

    data.scatterplot(
        x="NUMBER OF CYCLISTS INJURED",
        y="NUMBER OF PERSONS INJURED",
        title="Cyclists Injured vs All Persons Injured (Missing X)",
        width=900,
        height=520,
        show_legend=True,
        legend_title="Missingness",
        present_color="#2ca02c",
        missing_color="#d62728",
        point_size=7,
        axis_padding=0.06,
    ).show()

    data.scatterplot(
        x="ZIP CODE",
        y="NUMBER OF CYCLISTS INJURED",
        title="ZIP Code vs Cyclists Injured (Missing Y + Both)",
        width=960,
        height=520,
        background_color="#0b0b0b",
        text_color="#f2f2f2",
        present_color="#17becf",
        missing_color="#bcbd22",
        point_size=9,
        axis_padding=0.12,
    ).show()

    data.scatterplot(
        x="NUMBER OF PERSONS KILLED",
        y="NUMBER OF CYCLISTS KILLED",
        title="All Persons Killed vs Cyclists Killed (Missing Y)",
        width=950,
        height=520,
        show_legend=False,
        point_size=8,
        axis_padding=0.09,
    ).show()


if __name__ == "__main__":
    main()
