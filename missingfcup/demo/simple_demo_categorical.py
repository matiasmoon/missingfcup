from missingfcup import MissingObject, Panel
import pandas as pd

df = pd.DataFrame({
    "A": [1, None, 3, 4, None],  # numeric with missing entries
    "B": ["x", "y", None, "z", "w"],  # categorical with a missing
    "C": [10, 20, 30, None, 50],  # numeric with one missing value
})

item = MissingObject(df)

heatmap = item.heatmap(show_scale=True)
heatmap.show()

barchat = item.barchart()
barchat.show()

panel = Panel([
    item.heatmap(title="A"),
    item.heatmap(title="B"),
    item.heatmap(title="C"),
    item.heatmap(title="D"),
])

panel.show()
#panel.save("missing_panel")