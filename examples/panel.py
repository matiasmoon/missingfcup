import pandas as pd

from missingfcup import MissingData, Panel


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )
    data = MissingData(df)

    print("\n=== Panel Demo (5 Examples) ===")

    # ------------------------------------------------------------------
    # 1️⃣ Two plots
    # ------------------------------------------------------------------
    p1 = data.barchart_missing_count(title="Missing Count by Column")
    p2 = data.scatterplot_missingness(
        x="NUMBER OF PERSONS INJURED",
        y="NUMBER OF PERSONS KILLED",
        title="Injuries vs Fatalities",
    )
    Panel([p1, p2], title="Panel: 2 Plots").show()

    # ------------------------------------------------------------------
    # 2️⃣ Two plots (themed)
    # ------------------------------------------------------------------
    p3 = data.barchart_venn(
        selected_columns=["ZIP CODE", "BOROUGH", "OFF STREET NAME"],
        title="ZIP/Borough/Off-Street Patterns",
    )
    p4 = data.scatterplot_missingness(
        x="ZIP CODE",
        y="LATITUDE",
        title="ZIP Code vs Latitude",
        point_size=9,
    )
    Panel(
        [p3, p4],
        title="Panel: Location Patterns",
        background_color="#f8f5f0",
        text_color="#1b1b1b",
    ).show()

    # ------------------------------------------------------------------
    # 3️⃣ Three plots
    # ------------------------------------------------------------------
    p5 = data.barchart_missing_count(
        max_columns=12,
        title="Missing Count (Top 12)",
    )
    p6 = data.barchart_venn(
        selected_columns=["ON STREET NAME", "CROSS STREET NAME", "OFF STREET NAME"],
        title="Street Fields Patterns",
        value="percent",
    )
    p7 = data.scatterplot_missingness(
        x="NUMBER OF PEDESTRIANS INJURED",
        y="NUMBER OF PEDESTRIANS KILLED",
        title="Pedestrians: Injured vs Killed",
    )
    Panel([p5, p6, p7], title="Panel: 3 Plots").show()

    # ------------------------------------------------------------------
    # 4️⃣ Four plots
    # ------------------------------------------------------------------
    p8 = data.barchart_missing_count(
        selected_columns=[
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
            "CONTRIBUTING FACTOR VEHICLE 3",
        ],
        title="Contributing Factors (Counts)",
    )
    p9 = data.barchart_venn(
        selected_columns=[
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
            "CONTRIBUTING FACTOR VEHICLE 3",
        ],
        title="Contributing Factors (Patterns)",
    )
    p10 = data.scatterplot_missingness(
        x="NUMBER OF PERSONS INJURED",
        y="ZIP CODE",
        title="People Injured vs ZIP Code",
    )
    p11 = data.scatterplot_missingness(
        x="LATITUDE",
        y="LONGITUDE",
        title="Latitude vs Longitude",
    )
    Panel([p8, p9, p10, p11], title="Panel: 4 Plots").show()

    # ------------------------------------------------------------------
    # 5️⃣ Four plots (dark theme)
    # ------------------------------------------------------------------
    p12 = data.barchart_missing_count(
        max_columns=10,
        title="Missing Count (Top 10)",
        present_color="#17becf",
        missing_color="#bcbd22",
    )
    p13 = data.barchart_venn(
        selected_columns=["ZIP CODE", "BOROUGH", "ON STREET NAME"],
        title="Location Patterns",
        missing_color="#ff7f0e",
    )
    p14 = data.scatterplot_missingness(
        x="NUMBER OF PERSONS KILLED",
        y="NUMBER OF PEDESTRIANS KILLED",
        title="Fatalities: Persons vs Pedestrians",
        point_size=7,
    )
    p15 = data.scatterplot_missingness(
        x="NUMBER OF PERSONS INJURED",
        y="NUMBER OF PEDESTRIANS INJURED",
        title="Injuries: Persons vs Pedestrians",
        point_size=7,
    )
    Panel(
        [p12, p13, p14, p15],
        title="Panel: 4 Plots (Dark Theme)",
        background_color="#0b0b0b",
        text_color="#f2f2f2",
    ).show()


if __name__ == "__main__":
    main()
