import pandas as pd
from missingfcup_pkg import missing_matrix
import plotly.io as pio


# Load dataset
collisions = pd.read_csv(
    "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
)
print(f"Dataset shape: {collisions.shape}")

#import plotly.io as pio

#fig = missing_matrix(collisions)
#pio.show(fig)

# --- Default plot ---
#fig = missing_matrix(collisions)
#fig.show()

# --- Larger figure ---
fig = missing_matrix(collisions, max_cols=10, height=800, width=1200)
pio.show(fig)

#fig = missing_matrix(collisions, max_cols=50, height=800, width=1200)
#fig.show()

# --- Custom colors ---
fig = missing_matrix(
    collisions, 
    height=800, width=1200, 
    present_color="#1f77b4", missing_color="#ff7f0e", 
    show_scale=True, 
    title="Missing Values in  the NYPD Motor Vehicle Collisions Dataset"
)
pio.show(fig)

# --- Subset columns ---
cols_subset = ["BOROUGH", "CONTRIBUTING_FACTOR_VEHICLE_1", "VEHICLE_TYPE_CODE_1", "DATE"]
fig = missing_matrix(collisions, selected_columns=cols_subset, height=500, width=700)
pio.show(fig)

# --- Minimal view ---
fig = missing_matrix(collisions, show_hover=False, show_scale=False)
pio.show(fig)