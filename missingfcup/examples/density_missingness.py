import pandas as pd

from missingfcup import MissingData


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )
    md = MissingData(df)

    print("\n=== Density Missingness Demo (10 Examples) ===")
    print("Question asked: does the distribution of X differ between rows")
    print("where Y is present vs. missing? Overlapping curves → MCAR-like,")
    print("diverging curves → MAR or MNAR signal.\n")

    # ------------------------------------------------------------------
    # 1️⃣ Very easy (minimal)
    # ------------------------------------------------------------------
    md.density_missingness(
        x="NUMBER OF PERSONS INJURED",
        color_by="ZIP CODE",
    ).show()

    # ------------------------------------------------------------------
    # 2️⃣-5️⃣ Simple with personalization
    # ------------------------------------------------------------------
    md.density_missingness(
        x="NUMBER OF PERSONS INJURED",
        color_by="ZIP CODE",
        title="Persons Injured — ZIP Code present vs. missing",
    ).show()

    md.density_missingness(
        x="NUMBER OF PERSONS INJURED",
        color_by="BOROUGH",
        title="Persons Injured — Borough present vs. missing",
        fill_opacity=0.3,
    ).show()

    md.density_missingness(
        x="NUMBER OF PERSONS INJURED",
        color_by="ON STREET NAME",
        title="Persons Injured — Street Name present vs. missing",
        present_color="#4C78A8",
        missing_color="#F58518",
        fill_opacity=0.5,
    ).show()

    md.density_missingness(
        x="NUMBER OF PERSONS KILLED",
        color_by="ZIP CODE",
        title="Persons Killed — ZIP Code present vs. missing",
        n_points=500,
        fill_opacity=0.35,
    ).show()

    # ------------------------------------------------------------------
    # 6️⃣-🔟 More complex
    # ------------------------------------------------------------------
    md.density_missingness(
        x="NUMBER OF PEDESTRIANS INJURED",
        color_by="OFF STREET NAME",
        title="Pedestrians Injured — Off-Street Name present vs. missing",
        n_points=400,
        width=900,
        height=480,
    ).show()

    md.density_missingness(
        x="NUMBER OF CYCLIST INJURED",
        color_by="CROSS STREET NAME",
        title="Cyclists Injured — Cross Street present vs. missing",
        n_points=400,
        fill_opacity=0.45,
        width=900,
        height=480,
    ).show()

    md.density_missingness(
        x="NUMBER OF PERSONS INJURED",
        color_by="VEHICLE TYPE CODE 2",
        title="Persons Injured — Vehicle Type Code 2 present vs. missing",
        present_color="#72B7B2",
        missing_color="#FF9DA7",
        fill_opacity=0.4,
        n_points=350,
        width=900,
        height=480,
    ).show()

    md.density_missingness(
        x="NUMBER OF MOTORIST INJURED",
        color_by="CONTRIBUTING FACTOR VEHICLE 2",
        title="Motorists Injured — Contributing Factor 2 present vs. missing",
        n_points=400,
        fill_opacity=0.4,
        width=950,
        height=500,
    ).show()

    md.density_missingness(
        x="NUMBER OF PERSONS INJURED",
        color_by="ZIP CODE",
        title="Persons Injured — ZIP Code (Dark Theme)",
        background_color="#0b0b0b",
        text_color="#f2f2f2",
        present_color="#17becf",
        missing_color="#e377c2",
        fill_opacity=0.4,
        n_points=400,
        width=900,
        height=500,
    ).show()


if __name__ == "__main__":
    main()
