import pandas as pd
from missingfcup.core.missingobject import MissingObject
import plotly.io as pio


try:
    # Load dataset
    collisions = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )
    print(f"Dataset shape: {collisions.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please check your internet connection and try again.")
    exit(1)

try:
    item = MissingObject(collisions)
    print(item.matrix)
    heatmap = item.heatmap()
    heatmap.show()
except Exception as e:
    print(f"Error creating figure: {e}")

#try:
#    fig = missing_matrix(
#        collisions, 
#        height=800, width=1200, 
#        present_color="#1f77b4", missing_color="#ff7f0e", 
#        show_scale=True, 
#        title="Missing Values in  the NYPD Motor Vehicle Collisions Dataset"
#    )
#    pio.show(fig)
#except Exception as e:
#    print(f"Error creating custom colors figure: {e}")

#try:
#    cols_subset = ["BOROUGH", "CONTRIBUTING FACTOR VEHICLE 1", "VEHICLE TYPE CODE 1", "DATE"]
#    fig = missing_matrix(collisions, selected_columns=cols_subset, height=500, width=700)
#    pio.show(fig)
#except Exception as e:
#    print(f"Error creating subset columns figure: {e}")