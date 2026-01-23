import pandas as pd
from missingfcup.core.MissingData import MissingData
from missingfcup.core.ViewMetadata import OrderBySpec, ViewMetadata, OrderType, NumericOrder

# Load airquality dataset (same one used in naniar examples)
df = pd.read_csv(
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv"
)

# Drop the row index column added by Rdatasets
df = df.drop(columns=["rownames"])

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


metadata = ViewMetadata(
    order_by=[
        OrderBySpec(
            column="Ozone",
            type=OrderType.NUMERIC,
            numeric_order=NumericOrder.ASC,
        )
    ]
)

md.heatmap(
    metadata=metadata,
    title="Missingness heatmap (rows ordered by Ozone)",
).show()

md.pattern_barchart(
    title="Number of rows with same missing patterns",
    max_patterns=10,
).show()