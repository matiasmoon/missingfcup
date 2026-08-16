"""Dataset-wide totals: present cells against missing cells.

Two bars, no per-column breakdown. Useful when the only question is how much of
the dataset is missing overall.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.totals(title="Present vs missing cells").show()
