"""Several plots combined into one figure.

Panel arranges plots in a grid. Give each plot its own title; the panel title sits
above the whole grid. Handy for putting an overview on a single screen.
"""

import missingfcup as mf
from missingfcup import Panel

df = mf.sample_data()
md = mf.MissingData(df)

panel = Panel(
    [
        md.matrix(title="Matrix"),
        md.bar(title="Missing counts"),
        md.rate(title="Missing rates"),
    ],
    title="Missing data overview",
)
panel.show()
