import pandas as pd
import numpy as np

from missingfcup.core.MissingData import MissingData

def main():
    df = pd.read_csv("https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv")
    data = MissingData(df)

    scatter = data.scatterplot(
        x="NUMBER OF PERSONS INJURED",
        y="NUMBER OF PERSONS KILLED",
        title="Injuries vs Fatalities",
    )
    scatter.show()

if __name__ == "__main__":
    main()
