import pandas as pd
import missingfcup.core.MissingData as MissingData

df = pd.read_csv("https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv")
#df.info()

md = MissingData(df)
md.barchart().show()

md.heatmap().show()

md.scatterplot(
    x="NUMBER OF PERSONS INJURED",
    y="NUMBER OF PERSONS KILLED",
    title="Injuries vs Deaths (colored by missingness)",
).show()