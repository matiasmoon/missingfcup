"""Demo: Missing data visualization on NYPD motor vehicle collisions dataset.

This script demonstrates the missing_matrix function on a real-world dataset
with various customization options, including color changes, column selection,
and different display modes.

The dataset is loaded from GitHub and visualized in multiple ways to showcase
different features of the missing_matrix function.
"""

import pandas as pd
from missingfcup.viz.missing_matrix import missing_matrix
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

#import plotly.io as pio

#fig = missing_matrix(collisions)
#pio.show(fig)

# --- Default plot ---
#fig = missing_matrix(collisions)
#fig.show()

# --- Larger figure ---
try:
    fig = missing_matrix(collisions, max_cols=10, height=800, width=1200)
    pio.show(fig)
except Exception as e:
    print(f"Error creating figure: {e}")

#fig = missing_matrix(collisions, max_cols=50, height=800, width=1200)
#fig.show()

# --- Custom colors ---
try:
    fig = missing_matrix(
        collisions, 
        height=800, width=1200, 
        present_color="#1f77b4", missing_color="#ff7f0e", 
        show_scale=True, 
        title="Missing Values in  the NYPD Motor Vehicle Collisions Dataset"
    )
    pio.show(fig)
except Exception as e:
    print(f"Error creating custom colors figure: {e}")

# --- Subset columns ---
try:
    cols_subset = ["BOROUGH", "CONTRIBUTING FACTOR VEHICLE 1", "VEHICLE TYPE CODE 1", "DATE"]
    fig = missing_matrix(collisions, selected_columns=cols_subset, height=500, width=700)
    pio.show(fig)
except Exception as e:
    print(f"Error creating subset columns figure: {e}")

# --- Minimal view ---
try:
    fig = missing_matrix(collisions, show_hover=False, show_scale=False)
    pio.show(fig)
except Exception as e:
    print(f"Error creating minimal view figure: {e}")