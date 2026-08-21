"""Per-column missing rate, as a fraction or a percentage.

Counts answer "how many"; rates answer "how much of the column". Rates are the
better choice when comparing columns of differing completeness.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.bar(measure="fraction", title="Missing rate per column").show()
# mf.bar(df, measure="fraction", title="Missing rate per column").show()

# The same numbers on a 0-100 scale.
md.bar(measure="percentage", title="Missing percentage per column").show()
# mf.bar(df, measure="percentage", title="Missing percentage per column").show()
