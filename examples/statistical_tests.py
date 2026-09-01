"""The three statistical tests, and what each one is for.

The plots make a case visually; these put a number on it. They print rather than draw,
so this script has no figure to show.

The order below is the order to use them in. Little's test asks whether the data is MCAR
at all. If it rejects, the other two ask the follow-up question the plots ask visually:
does one column's distribution depend on whether another column is missing?
"""

import missingfcup as mf

df = mf.sample_data()
md = mf.MissingData(df)

# Little's test compares the means across every distinct missingness pattern against one
# common mean. A small p rejects MCAR; it does not say which of MAR or MNAR replaces it.
# Read it as one input among several, not as a verdict -- notebooks/README.md records
# three datasets where it misleads.
print("Little's MCAR test")
print(md.littles_mcar_test().to_string(), "\n")

# Only numeric columns can be compared this way, and the test says so rather than
# silently dropping them. numeric_only=False is available for the rare frame where a
# non-numeric column is genuinely orderable.
print("restricted to two columns")
print(md.littles_mcar_test(columns=["age", "income"]).to_string(), "\n")

# Mann-Whitney compares the observed values of x between the rows where `by` is present
# and the rows where it is missing. It is a rank test, so it assumes no particular
# distribution shape, and it is the statistical counterpart of density() and boxplot().
print("Mann-Whitney: does income depend on whether age is missing")
print(md.mann_whitney_test(x="income", by="age").to_string(), "\n")

# The KS test asks the same question and measures a different thing, which decides which
# one finds a given relationship. Mann-Whitney detects that one group sits *above* the
# other. KS compares the two distribution functions at every point and reports their
# largest gap, so it also detects a difference with no direction -- missingness drawn
# from both tails of a column at once, where the two groups share a centre and a rank
# test sees nothing.
print("Kolmogorov-Smirnov: the same pair, measured without a direction")
print(md.ks_test(x="income", by="age").to_string(), "\n")

# Both return the group medians alongside the p-value, so a significant result can be
# read as a magnitude rather than only as a yes.
result = md.mann_whitney_test(x="income", by="age")
print(f"median income where age is present: {result['median_present']}")
print(f"median income where age is missing: {result['median_missing']}")
print(f"significant at 0.05: {result['significant']}")
