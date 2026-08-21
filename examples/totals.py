"""Dataset-wide totals: present cells against missing cells.

Two bars, no per-column breakdown. Useful when the only question is how much of
the dataset is missing overall.

Commented are the flat functions that produce the same plot as object calls.
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.totals(title="Present vs missing cells").show()
# mf.totals(df, title="Present vs missing cells").show()

# The two bars carry their count and share above them by default. Turning that off
# leaves the comparison to the bar heights, which is what a slide usually wants;
# the numbers are still on the hover.
md.totals(
    show_values=False,
    missing_color="#B22222",
    present_color="#2E8B57",
    title="How complete is the dataset",
).show()
# mf.totals(df, show_values=False, missing_color="#B22222",
#           present_color="#2E8B57", title="How complete is the dataset").show()
