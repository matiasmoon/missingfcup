# To run this file, you need to have the `missingfcup` package installed.

from missingfcup import MissingData, Panel
import pandas as pd
#from missingfcup.plots import Barchart # This import should fail

df = pd.DataFrame({
    "a": [1, None, 3],
    "b": [None, None, 2],
})

md = MissingData(df)

md.barchart().show()