"""Per-column missing rate, as a fraction or a percentage.

Counts answer "how many"; rates answer "how much of the column". Rates are the
better choice when comparing columns of differing completeness.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.bar(measure="rate", title="Missing rate per column").show()

# The same numbers on a 0-100 scale.
md.bar(measure="rate", scale="percentage", title="Missing percentage per column").show()
