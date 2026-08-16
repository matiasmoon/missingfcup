"""The same comparison as density, drawn as boxes or violins.

Box plots make the medians and quartiles easy to compare; violins also show the
shape of each distribution. Both answer: does this column's distribution shift
depending on whether another column is missing?
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

md.boxplot(x="income", color_by="age", title="income by age missingness").show()

md.boxplot(x="income", color_by="age", plot_type="violin", title="Same split, as violins").show()
