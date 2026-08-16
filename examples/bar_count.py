"""Per-column missing counts.

The default bar chart. Use show_both=True to stack present against missing, so the
column height becomes the row count and the split is visible at a glance.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.bar(title="Missing values per column").show()

# Present and missing stacked together.
md.bar(show_both=True, title="Present vs missing per column").show()
