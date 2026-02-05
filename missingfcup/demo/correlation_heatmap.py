import pandas as pd
from missingfcup.core.MissingData import MissingData


def main():
    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )

    # ------------------------------------------------------------------
    # Create MissingData object
    # ------------------------------------------------------------------
    md = MissingData(df)

    print("=== Missingness Correlation Heatmap Demo ===")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Overall missingness: {md.total_missingness:.2%}")

    # ------------------------------------------------------------------
    # Missingness correlation heatmap
    # ------------------------------------------------------------------
    heatmap = md.missingness_correlation_heatmap(
        title="Missingness Correlation Heatmap",
        show_values=True,
    )

    heatmap.show()


if __name__ == "__main__":
    main()
