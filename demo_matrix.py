import pandas as pd
from missingfcup_pkg.matrix import missing_matrix

df = pd.DataFrame({
    "A": [1, None, 3, 4, None],
    "B": ["x", "y", None, "z", "w"],
    "C": [10, 20, 30, None, 50],
})

fig = missing_matrix(df)  # super simple, defaults used
fig.show()

#fig = missing_matrix(
#    df,
#    max_cols=100,
#    height=800,
#    width=1200,
#    present_color="#1f77b4",
#    missing_color="#ff7f0e",
#    selected_columns=["A", "C"],
#    show_hover=True,
#    show_scale=True
#)
#fig.show()
