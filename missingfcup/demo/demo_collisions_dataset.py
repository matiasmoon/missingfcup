"""
Demo: NYC Motor Vehicle Collisions Dataset - Comprehensive Analysis

This demo showcases a real-world dataset analysis using the NYC collision data.
Demonstrates all missingfcup features with practical examples including:
- Pattern discovery through grouping
- Binary vs Completeness visualization modes
- High missingness filtering for cleaner analysis
- Side-by-side comparisons for better insights
"""

import pandas as pd
from missingfcup import MissingObject
from plotly.subplots import make_subplots

print("="*80)
print("NYC MOTOR VEHICLE COLLISIONS DATASET - MISSING DATA ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATASET
# ============================================================================
print("\nLoading NYC collision dataset from GitHub...")

try:
    collisions = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {collisions.shape[0]:,} rows × {collisions.shape[1]} columns")
    print(f"  Columns: {list(collisions.columns)}")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    print("  Please check your internet connection and try again.")
    exit(1)

missing = MissingObject(collisions)

# ============================================================================
# SECTION 1: INITIAL EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: INITIAL EXPLORATION")
print("="*80)

# Example 1.1: Overview of missing data patterns
print("\n1.1 Basic Heatmap - Overview of Missing Data Patterns")
print("    First look at the dataset's missing value structure")

heatmap_basic = missing.heatmap(
    title="NYC Collisions: Missing Data Overview",
    show_colorscale_legend=True,
    figure_size_pixels=(1200, 800)
)
heatmap_basic.show()

# Example 1.2: With high missingness filtering (NEW FEATURE!)
print("\n1.2 Comparing: Default Filtering vs No Filtering")
print("    See the impact of excluding high-missingness columns")

heatmap_filtered = missing.heatmap(
    ignore_high_missingness=True,  # Default behavior
    high_missingness_threshold=0.9,
    title="Filtered View (≥90% missing excluded)",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 700)
)

heatmap_unfiltered = missing.heatmap(
    ignore_high_missingness=False,  # Include all columns
    title="Full View (All columns)",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 700)
)

fig_compare_1 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Filtered (Clean Analysis)", "Unfiltered (All Data)"),
    horizontal_spacing=0.12
)

for trace in heatmap_filtered.fig.data:
    fig_compare_1.add_trace(trace, row=1, col=1)
for trace in heatmap_unfiltered.fig.data:
    fig_compare_1.add_trace(trace, row=1, col=2)

fig_compare_1.update_layout(
    title_text="Impact of High Missingness Filtering on Real Dataset",
    height=800,
    showlegend=False
)
fig_compare_1.show()

# ============================================================================
# SECTION 2: GROUPING BY BOROUGH
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: GROUPING BY BOROUGH")
print("="*80)
print("Analyze if missing data patterns differ across NYC boroughs")

# Example 2.1: Binary mode - see specific missing cells by borough
print("\n2.1 Binary Mode: Individual Missing Values by Borough")

heatmap_borough_binary = missing.heatmap(
    group_by="BOROUGH",
    group_by_mode="binary",
    title="Borough Analysis: Binary Mode (Cell-Level)",
    show_colorscale_legend=True,
    figure_size_pixels=(1200, 900)
)
heatmap_borough_binary.show()

# Example 2.2: Completeness mode - overall data quality by borough
print("\n2.2 Completeness Mode: Data Quality by Borough")
print("    NEW FEATURE: Shows row-level completeness percentage")

heatmap_borough_comp = missing.heatmap(
    group_by="BOROUGH",
    group_by_mode="completeness",  # NEW: Show completeness %
    title="Borough Analysis: Completeness Mode (Row-Level %)",
    show_colorscale_legend=True,
    figure_size_pixels=(1200, 900)
)
heatmap_borough_comp.show()

# Example 2.3: Side-by-side comparison of both modes
print("\n2.3 Comparing: Binary vs Completeness Mode for Borough Analysis")

fig_compare_2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Binary: See Individual Patterns", "Completeness: See Overall Quality"),
    horizontal_spacing=0.12
)

for trace in heatmap_borough_binary.fig.data:
    fig_compare_2.add_trace(trace, row=1, col=1)
for trace in heatmap_borough_comp.fig.data:
    fig_compare_2.add_trace(trace, row=1, col=2)

fig_compare_2.update_layout(
    title_text="Borough Analysis: Binary vs Completeness Visualization",
    height=900,
    showlegend=False
)
fig_compare_2.show()

# ============================================================================
# SECTION 3: FOCUSING ON SPECIFIC COLUMNS
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: FOCUSED ANALYSIS ON KEY COLUMNS")
print("="*80)

# Example 3.1: Select specific columns of interest
print("\n3.1 Analyzing Key Contributing Factor Columns")

key_columns = [
    "BOROUGH",
    "CONTRIBUTING FACTOR VEHICLE 1",
    "CONTRIBUTING FACTOR VEHICLE 2",
    "VEHICLE TYPE CODE 1",
    "VEHICLE TYPE CODE 2",
]

heatmap_key_cols = missing.heatmap(
    selected_columns=key_columns,
    group_by="BOROUGH",
    group_by_mode="binary",
    title="Key Columns Analysis by Borough",
    show_colorscale_legend=True,
    figure_size_pixels=(1000, 800)
)
heatmap_key_cols.show()

# Example 3.2: Most complete vs least complete columns
print("\n3.2 Comparing: Most Complete vs Least Complete Columns")

heatmap_most = missing.heatmap(
    completeness_mode="most",
    max_columns_by_completeness=10,
    group_by="BOROUGH",
    group_by_mode="completeness",
    title="10 Most Complete Columns",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 800)
)

heatmap_least = missing.heatmap(
    completeness_mode="least",
    max_columns_by_completeness=10,
    ignore_high_missingness=False,  # Include even sparse columns
    group_by="BOROUGH",
    group_by_mode="completeness",
    title="10 Least Complete Columns",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 800)
)

fig_compare_3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Highest Quality Columns", "Lowest Quality Columns"),
    horizontal_spacing=0.12
)

for trace in heatmap_most.fig.data:
    fig_compare_3.add_trace(trace, row=1, col=1)
for trace in heatmap_least.fig.data:
    fig_compare_3.add_trace(trace, row=1, col=2)

fig_compare_3.update_layout(
    title_text="Data Quality Spectrum: Most vs Least Complete Columns",
    height=900,
    showlegend=False
)
fig_compare_3.show()

# ============================================================================
# SECTION 4: ADVANCED THRESHOLD TUNING
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: FINE-TUNING MISSINGNESS THRESHOLDS")
print("="*80)
print("NEW FEATURE: Customize which columns to exclude based on missingness level")

# Example 4.1: Compare different threshold levels
print("\n4.1 Comparing Different Missingness Thresholds")

thresholds = [0.7, 0.8, 0.9]

fig_thresholds = make_subplots(
    rows=1, cols=3,
    subplot_titles=(f"≥70% excluded", f"≥80% excluded", f"≥90% excluded"),
    horizontal_spacing=0.15
)

for idx, threshold in enumerate(thresholds, 1):
    heatmap_thresh = missing.heatmap(
        high_missingness_threshold=threshold,
        group_by="BOROUGH",
        group_by_mode="completeness",
        show_colorscale_legend=True,
        figure_size_pixels=(600, 700)
    )
    for trace in heatmap_thresh.fig.data:
        fig_thresholds.add_trace(trace, row=1, col=idx)

fig_thresholds.update_layout(
    title_text="Fine-Tuning: Impact of Different Missingness Thresholds on NYC Data",
    height=800,
    width=1600,
    showlegend=False
)
fig_thresholds.show()

# ============================================================================
# SECTION 5: COMPLETENESS THRESHOLD ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: ANALYZING COLUMNS BY COMPLETENESS THRESHOLD")
print("="*80)

# Example 5.1: Only show columns that are at least 50% complete
print("\n5.1 Columns with ≥50% Completeness")

heatmap_50 = missing.heatmap(
    completeness_mode="most",
    completeness_threshold=0.5,
    group_by="BOROUGH",
    group_by_mode="binary",
    title="Columns with ≥50% Complete Data",
    show_colorscale_legend=True,
    figure_size_pixels=(1200, 800)
)
heatmap_50.show()

# Example 5.2: Show problematic columns (less than 30% complete)
print("\n5.2 Problematic Columns with <30% Completeness")

try:
    heatmap_problematic = missing.heatmap(
        completeness_mode="least",
        completeness_threshold=0.3,
        ignore_high_missingness=False,
        group_by="BOROUGH",
        group_by_mode="completeness",
        title="Problematic Columns (<30% Complete)",
        show_colorscale_legend=True,
        figure_size_pixels=(1200, 800)
    )
    heatmap_problematic.show()
except ValueError as e:
    print(f"    Note: {e}")

# ============================================================================
# SECTION 6: CUSTOM VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: CUSTOM COLOR SCHEMES")
print("="*80)

# Example 6.1: Custom colors for better visibility
print("\n6.1 Custom Color Scheme for Publication")

heatmap_custom = missing.heatmap(
    group_by="BOROUGH",
    group_by_mode="binary",
    present_color="#1f77b4",  # Professional blue
    missing_color="#ff7f0e",  # Professional orange
    title="NYC Collisions: Custom Color Scheme",
    show_colorscale_legend=True,
    figure_size_pixels=(1200, 900)
)
heatmap_custom.show()

# ============================================================================
# SECTION 7: COMBINED INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: COMPREHENSIVE ANALYSIS - ALL FEATURES COMBINED")
print("="*80)

# Example 7.1: Ultimate comparison - 4 different configurations
print("\n7.1 Four-Panel Comprehensive Comparison")

# Create 4 different views
heatmap_1 = missing.heatmap(
    group_by="BOROUGH",
    group_by_mode="binary",
    title="Binary + Filtered",
    show_colorscale_legend=True,
    figure_size_pixels=(600, 600)
)

heatmap_2 = missing.heatmap(
    group_by="BOROUGH",
    group_by_mode="completeness",
    title="Completeness + Filtered",
    show_colorscale_legend=True,
    figure_size_pixels=(600, 600)
)

heatmap_3 = missing.heatmap(
    group_by="BOROUGH",
    group_by_mode="binary",
    ignore_high_missingness=False,
    title="Binary + Unfiltered",
    show_colorscale_legend=True,
    figure_size_pixels=(600, 600)
)

heatmap_4 = missing.heatmap(
    group_by="BOROUGH",
    group_by_mode="completeness",
    ignore_high_missingness=False,
    title="Completeness + Unfiltered",
    show_colorscale_legend=True,
    figure_size_pixels=(600, 600)
)

# Create 2x2 grid
fig_final = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Binary Mode (Filtered)",
        "Completeness Mode (Filtered)",
        "Binary Mode (All Columns)",
        "Completeness Mode (All Columns)"
    ),
    horizontal_spacing=0.12,
    vertical_spacing=0.15
)

for trace in heatmap_1.fig.data:
    fig_final.add_trace(trace, row=1, col=1)
for trace in heatmap_2.fig.data:
    fig_final.add_trace(trace, row=1, col=2)
for trace in heatmap_3.fig.data:
    fig_final.add_trace(trace, row=2, col=1)
for trace in heatmap_4.fig.data:
    fig_final.add_trace(trace, row=2, col=2)

fig_final.update_layout(
    title_text="NYC Collisions: Comprehensive 4-Panel Analysis",
    height=1200,
    width=1600,
    showlegend=False
)
fig_final.show()

# ============================================================================
# SUMMARY AND INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)
print(f"""
Dataset Overview:
• Total Records: {collisions.shape[0]:,}
• Total Columns: {collisions.shape[1]}
• Boroughs Analyzed: {collisions['BOROUGH'].nunique()}

Key Features Demonstrated:

1. HIGH MISSINGNESS FILTERING (NEW!)
   ✓ Automatically excludes columns with ≥90% missing values
   ✓ Customizable threshold (70%, 80%, 90%, etc.)
   ✓ Keeps analysis focused on meaningful columns

2. VISUALIZATION MODES (NEW!)
   ✓ Binary Mode: See individual missing cells
     - Best for: Identifying specific data gaps
     - Use case: Data quality audits

   ✓ Completeness Mode: See row-level completeness %
     - Best for: Comparing overall data quality
     - Use case: Group comparisons, quality reports

3. GROUPING & SORTING
   ✓ Group by BOROUGH to reveal geographic patterns
   ✓ Grouping columns highlighted with orange borders
   ✓ Rows sorted by group for pattern visibility

4. COLUMN SELECTION
   ✓ Focus on specific columns of interest
   ✓ Select most/least complete columns
   ✓ Set completeness thresholds

Recommended Workflows:

📊 Exploratory Analysis:
   → Use Completeness Mode + Borough grouping
   → Start with default filtering (≥90%)
   → Identify patterns across boroughs

🔍 Detailed Investigation:
   → Use Binary Mode for specific columns
   → Lower threshold to include more columns
   → Focus on problematic areas

📈 Quality Reporting:
   → Compare filtered vs unfiltered views
   → Use custom color schemes
   → Create multi-panel comparisons

Next Steps:
• Investigate why certain boroughs have more missing data
• Analyze patterns in contributing factors
• Use completeness mode to prioritize data collection efforts
• Consider temporal patterns (add date-based grouping)
""")

print("="*80)
print("Demo completed! Check the visualizations for insights.")
print("="*80)
