import pandas as pd
from missingfcup.core.MissingData import MissingData


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )

    md = MissingData(df)

    heatmap = md.column_missing_rate_heatmap()

    print(heatmap.fig)

    heatmap.show()

    #heatmap.save("column_missing_rate_heatmap.html")

if __name__ == "__main__":
    main()