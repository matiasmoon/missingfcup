"""Missing rate drawn as a single coloured strip.

One row of cells, one per column, shaded by missing rate. This stays readable when
a dataset has many columns, where a bar chart would get crowded.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.rate(title="Missing rate per column").show()

md.rate(scale="percentage", title="Missing percentage per column").show()
