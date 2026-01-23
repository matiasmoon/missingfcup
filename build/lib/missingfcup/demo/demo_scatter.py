import pandas as pd
from missingfcup.core.MissingData import MissingData

# Load airquality dataset (same one used in naniar examples)
df = pd.read_csv(
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv"
)

# Drop the row index column added by Rdatasets
#df = df.drop(columns=["Unnamed: 0"])

md = MissingData(df)

# Optional diagnostics
md.barchart(title="Missing values per variable").show()
md.heatmap(title="Missingness heatmap").show()

# naniar-style scatterplot with missing offsets
md.scatterplot(
    x="Solar.R",
    y="Ozone",
    title="Ozone vs Solar Radiation (missing shown as offsets)",
).show()
