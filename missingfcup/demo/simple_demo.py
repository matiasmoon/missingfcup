from missingfcup import MissingObject, Panel
import pandas as pd

df = pd.DataFrame({
    "A": [1, None, 3],
    "B": [None, None, 6],
    "C": [7, 8, None],
})

item = MissingObject(df)

heatmap = item.heatmap(show_colorscale_legend=True)
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