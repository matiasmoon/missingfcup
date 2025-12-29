"""
Demo: Comprehensive Guide to Grouping and Visualization Features

This demo showcases all features of the missingfcup package including:
- Group by functionality with different sorting modes
- Binary vs Completeness visualization modes
- High missingness filtering
- Side-by-side comparisons using panels
"""

from missingfcup import MissingObject
import pandas as pd
from plotly.subplots import make_subplots

print("="*80)
print("MISSINGFCUP - COMPREHENSIVE FEATURE DEMONSTRATION")
print("="*80)

# ============================================================================
# SECTION 1: BASIC GROUPING FEATURES
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: BASIC GROUPING & SORTING")
print("="*80)

# Create sample data with various patterns
df = pd.DataFrame({
    'age': [25, 30, 22, 35, 28, 32, 29, 26, 31, 27],
    'income': [50000, None, 45000, None, 52000, 55000, None, 48000, None, 51000],
    'education': ['High', 'Med', 'Low', 'High', 'Med', 'High', 'Low', 'Med', 'High', 'Low'],
    'city': ['NYC', 'LA', 'NYC', 'LA', 'NYC', 'LA', 'NYC', 'LA', 'NYC', 'LA'],
    'score': [None, 85, 90, None, 88, None, 92, 86, None, 89],
    'bonus': [5000, None, None, 8000, None, 6000, None, None, 7000, 1000],
})

missing = MissingObject(df)

# Example 1: No grouping vs Grouped by age
print("\n1.1 Comparing: No Grouping vs Grouped by Age")
print("    Shows how grouping reveals patterns in missing data")

heatmap_no_group = missing.heatmap(
    title="No Grouping (Original Order)",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 500)
)

heatmap_by_age = missing.heatmap(
    group_by="age",
    group_direction="ascending",
    title="Grouped by Age (Ascending)",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 500)
)

# Create side-by-side comparison
from plotly.subplots import make_subplots
fig_comparison_1 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("No Grouping", "Grouped by Age"),
    horizontal_spacing=0.12
)

# Add traces from both heatmaps
for trace in heatmap_no_group.fig.data:
    fig_comparison_1.add_trace(trace, row=1, col=1)
for trace in heatmap_by_age.fig.data:
    fig_comparison_1.add_trace(trace, row=1, col=2)

fig_comparison_1.update_layout(
    title_text="Comparison: Impact of Grouping on Pattern Visibility",
    height=600,
    showlegend=False
)
fig_comparison_1.show()

# Example 2: Ascending vs Descending
print("\n1.2 Comparing: Ascending vs Descending Sort Order")

heatmap_asc = missing.heatmap(
    group_by="age",
    group_direction="ascending",
    title="Age: Ascending (22→35)",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 500)
)

heatmap_desc = missing.heatmap(
    group_by="age",
    group_direction="descending",
    title="Age: Descending (35→22)",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 500)
)

fig_comparison_2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Ascending Order", "Descending Order"),
    horizontal_spacing=0.12
)

for trace in heatmap_asc.fig.data:
    fig_comparison_2.add_trace(trace, row=1, col=1)
for trace in heatmap_desc.fig.data:
    fig_comparison_2.add_trace(trace, row=1, col=2)

fig_comparison_2.update_layout(
    title_text="Comparison: Sort Direction Impact",
    height=600,
    showlegend=False
)
fig_comparison_2.show()

# Example 3: Categorical grouping with custom order
print("\n1.3 Categorical Grouping with Custom Order")

heatmap_edu = missing.heatmap(
    group_by="education",
    group_categories=["Low", "Med", "High"],
    title="Grouped by Education (Low → Med → High)",
    show_colorscale_legend=True,
    figure_size_pixels=(900, 500)
)
heatmap_edu.show()

# Example 4: Multi-column grouping
print("\n1.4 Multi-Column Grouping (City → Age)")

heatmap_multi = missing.heatmap(
    group_by=["city", "age"],
    group_direction=["ascending", "ascending"],
    title="Grouped by City, then Age",
    show_colorscale_legend=True,
    figure_size_pixels=(1000, 600)
)
heatmap_multi.show()

# ============================================================================
# SECTION 2: BINARY VS COMPLETENESS MODE (NEW FEATURE!)
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: BINARY vs COMPLETENESS VISUALIZATION MODES")
print("="*80)
print("NEW FEATURE: Choose how to visualize missing data patterns")

# Example 5: Binary vs Completeness side-by-side
print("\n2.1 Comparing: Binary vs Completeness Mode")
print("    Binary: Shows individual cell presence/absence")
print("    Completeness: Shows row-level completeness percentage")

heatmap_binary = missing.heatmap(
    group_by="education",
    group_categories=["Low", "Med", "High"],
    group_by_mode="binary",  # Default: individual cells
    title="Binary Mode (Cell-Level)",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 500)
)

heatmap_completeness = missing.heatmap(
    group_by="education",
    group_categories=["Low", "Med", "High"],
    group_by_mode="completeness",  # NEW: row completeness
    title="Completeness Mode (Row-Level %)",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 500)
)

fig_comparison_3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Binary Mode: Present/Missing per Cell",
                    "Completeness Mode: Row Completeness %"),
    horizontal_spacing=0.12
)

for trace in heatmap_binary.fig.data:
    fig_comparison_3.add_trace(trace, row=1, col=1)
for trace in heatmap_completeness.fig.data:
    fig_comparison_3.add_trace(trace, row=1, col=2)

fig_comparison_3.update_layout(
    title_text="NEW FEATURE: Binary vs Completeness Visualization Modes",
    height=600,
    showlegend=False
)
fig_comparison_3.show()

# Example 6: Completeness mode with different groupings
print("\n2.2 Completeness Mode with Different Groupings")
print("    Shows how row completeness varies across different groups")

heatmap_comp_age = missing.heatmap(
    group_by="age",
    group_direction="ascending",
    group_by_mode="completeness",
    title="Completeness by Age",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 500)
)

heatmap_comp_city = missing.heatmap(
    group_by="city",
    group_by_mode="completeness",
    title="Completeness by City",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 500)
)

fig_comparison_4 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("By Age Group", "By City Group"),
    horizontal_spacing=0.12
)

for trace in heatmap_comp_age.fig.data:
    fig_comparison_4.add_trace(trace, row=1, col=1)
for trace in heatmap_comp_city.fig.data:
    fig_comparison_4.add_trace(trace, row=1, col=2)

fig_comparison_4.update_layout(
    title_text="Completeness Mode: Comparing Different Group Variables",
    height=600,
    showlegend=False
)
fig_comparison_4.show()

# ============================================================================
# SECTION 3: HIGH MISSINGNESS FILTERING (NEW FEATURE!)
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: HIGH MISSINGNESS FILTERING")
print("="*80)
print("NEW FEATURE: Control how columns with high missingness are handled")

# Create a dataset with columns that have varying levels of missingness
df_with_sparse = df.copy()
df_with_sparse['mostly_missing'] = [None] * 9 + [100]     # 90% missing
df_with_sparse['very_sparse'] = [None] * 9 + [200]        # 90% missing
df_with_sparse['almost_empty'] = [None, None] + [50] + [None] * 7  # 90% missing
df_with_sparse['completely_empty'] = [None] * 10          # 100% missing

missing_sparse = MissingObject(df_with_sparse)

# Example 7: Three filtering strategies side-by-side
print("\n3.1 Comparing: Different Filtering Strategies")
print("    a) Default: Exclude ≥90% missing (recommended)")
print("    b) Include all columns")
print("    c) Custom: Exclude ≥80% missing")

heatmap_default = missing_sparse.heatmap(
    title="Default Filter (≥90% excluded)",
    show_colorscale_legend=True,
    figure_size_pixels=(600, 500)
)

heatmap_all = missing_sparse.heatmap(
    ignore_high_missingness=False,
    title="No Filter (All columns)",
    show_colorscale_legend=True,
    figure_size_pixels=(600, 500)
)

heatmap_80 = missing_sparse.heatmap(
    high_missingness_threshold=0.8,
    title="Custom Filter (≥80% excluded)",
    show_colorscale_legend=True,
    figure_size_pixels=(600, 500)
)

# Create 3-panel comparison
fig_comparison_5 = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Default (≥90%)", "No Filter", "Custom (≥80%)"),
    horizontal_spacing=0.15
)

for trace in heatmap_default.fig.data:
    fig_comparison_5.add_trace(trace, row=1, col=1)
for trace in heatmap_all.fig.data:
    fig_comparison_5.add_trace(trace, row=1, col=2)
for trace in heatmap_80.fig.data:
    fig_comparison_5.add_trace(trace, row=1, col=3)

fig_comparison_5.update_layout(
    title_text="NEW FEATURE: High Missingness Filtering Strategies",
    height=600,
    width=1400,
    showlegend=False
)
fig_comparison_5.show()

# ============================================================================
# SECTION 4: COMBINING ALL FEATURES
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: COMBINING FEATURES FOR POWERFUL INSIGHTS")
print("="*80)

# Example 8: All features combined
print("\n4.1 Combining: Grouping + Completeness + Filtering")

# Create a richer dataset
df_rich = pd.DataFrame({
    'department': ['Sales', 'Eng', 'Sales', 'HR', 'Eng', 'HR', 'Sales', 'Eng', 'HR', 'Sales',
                   'Eng', 'HR', 'Sales', 'Eng', 'HR', 'Sales', 'Eng', 'HR', 'Sales', 'Eng'],
    'experience': [1, 5, 2, 8, 3, 4, 6, 2, 7, 3, 4, 5, 1, 6, 3, 8, 2, 5, 4, 7],
    'salary': [45000, 85000, None, 65000, None, 58000, 72000, None, 70000, 48000,
               82000, None, 46000, 88000, 59000, None, 80000, 67000, None, 84000],
    'bonus': [None, 5000, 1000, None, 3000, None, 4000, 2000, None, None,
              4500, 2500, None, 5500, None, 3500, None, 2800, 3000, None],
    'stock_options': [None] * 15 + [10000, None, 8000, None, 12000],  # 75% missing
    'commission': [2000, None, 1500, None, None, None, 3000, None, None, 2500,
                   None, None, 1800, None, None, 4000, None, None, 2200, None],
    'rarely_filled': [None] * 18 + [100, None],  # 95% missing
})

missing_rich = MissingObject(df_rich)

heatmap_combo1 = missing_rich.heatmap(
    group_by="department",
    group_categories=["Sales", "Eng", "HR"],
    group_by_mode="binary",
    title="Binary Mode + Grouping + Default Filter",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 700)
)

heatmap_combo2 = missing_rich.heatmap(
    group_by="department",
    group_categories=["Sales", "Eng", "HR"],
    group_by_mode="completeness",
    title="Completeness Mode + Grouping + Default Filter",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 700)
)

fig_comparison_6 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Binary: See Individual Patterns",
                    "Completeness: See Overall Health"),
    horizontal_spacing=0.12
)

for trace in heatmap_combo1.fig.data:
    fig_comparison_6.add_trace(trace, row=1, col=1)
for trace in heatmap_combo2.fig.data:
    fig_comparison_6.add_trace(trace, row=1, col=2)

fig_comparison_6.update_layout(
    title_text="COMBINED: Grouping + Modes + Filtering by Department",
    height=800,
    showlegend=False
)
fig_comparison_6.show()

# Example 9: Show impact of including vs excluding high missingness
print("\n4.2 Impact of Including High Missingness Columns")

heatmap_clean = missing_rich.heatmap(
    group_by="department",
    group_categories=["Sales", "Eng", "HR"],
    group_by_mode="completeness",
    ignore_high_missingness=True,
    high_missingness_threshold=0.9,
    title="Clean View (≥90% missing excluded)",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 700)
)

heatmap_full = missing_rich.heatmap(
    group_by="department",
    group_categories=["Sales", "Eng", "HR"],
    group_by_mode="completeness",
    ignore_high_missingness=False,
    title="Full View (All columns included)",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 700)
)

fig_comparison_7 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Filtered (Clean Analysis)",
                    "Unfiltered (Includes Sparse Columns)"),
    horizontal_spacing=0.12
)

for trace in heatmap_clean.fig.data:
    fig_comparison_7.add_trace(trace, row=1, col=1)
for trace in heatmap_full.fig.data:
    fig_comparison_7.add_trace(trace, row=1, col=2)

fig_comparison_7.update_layout(
    title_text="Impact of High Missingness Filtering on Analysis Clarity",
    height=800,
    showlegend=False
)
fig_comparison_7.show()

# ============================================================================
# SECTION 5: ADVANCED USE CASES
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: ADVANCED USE CASES")
print("="*80)

# Example 10: Multi-level grouping with completeness
print("\n5.1 Multi-Level Grouping with Completeness Mode")

heatmap_multi_comp = missing_rich.heatmap(
    group_by=["department", "experience"],
    group_categories=[["Sales", "Eng", "HR"], None],
    group_direction=["ascending", "ascending"],
    group_by_mode="completeness",
    title="Department → Experience: Completeness View",
    show_colorscale_legend=True,
    figure_size_pixels=(1000, 800)
)
heatmap_multi_comp.show()

# Example 11: Comparison across different thresholds
print("\n5.2 Fine-Tuning Missingness Threshold")

thresholds = [0.7, 0.8, 0.9]
print(f"    Comparing thresholds: {thresholds}")

fig_thresholds = make_subplots(
    rows=1, cols=3,
    subplot_titles=(f"≥70% excluded", f"≥80% excluded", f"≥90% excluded"),
    horizontal_spacing=0.15
)

for idx, threshold in enumerate(thresholds, 1):
    heatmap_thresh = missing_rich.heatmap(
        group_by="department",
        group_categories=["Sales", "Eng", "HR"],
        group_by_mode="binary",
        high_missingness_threshold=threshold,
        show_colorscale_legend=True,
        figure_size_pixels=(500, 600)
    )
    for trace in heatmap_thresh.fig.data:
        fig_thresholds.add_trace(trace, row=1, col=idx)

fig_thresholds.update_layout(
    title_text="Fine-Tuning: Impact of Different Missingness Thresholds",
    height=700,
    width=1400,
    showlegend=False
)
fig_thresholds.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FEATURE SUMMARY")
print("="*80)
print("""
✓ GROUPING FEATURES:
  • group_by: Sort rows by column values (single or multiple columns)
  • group_direction: "ascending" or "descending"
  • group_categories: Custom order for categorical columns
  • Supports up to 2 grouping columns
  • Grouping columns are highlighted with orange borders

✓ VISUALIZATION MODES (NEW!):
  • group_by_mode="binary" (default):
    - Shows presence (green) or absence (red) for each cell
    - Best for seeing individual missing value patterns

  • group_by_mode="completeness" (NEW!):
    - Shows row-level completeness as percentage (0-100%)
    - Color gradient: Red (0%) → Yellow (50%) → Green (100%)
    - Best for comparing overall data quality across groups

✓ HIGH MISSINGNESS FILTERING (NEW!):
  • ignore_high_missingness=True (default):
    - Automatically excludes columns with extreme missingness
    - Keeps analysis focused on meaningful columns

  • high_missingness_threshold=0.9 (default):
    - Customizable threshold (0 to 1)
    - 0.9 = exclude columns with ≥90% missing values
    - Adjust based on your data quality needs

✓ BEST PRACTICES:
  1. Start with default filtering to see clean patterns
  2. Use binary mode to identify specific missing cells
  3. Use completeness mode to compare group data quality
  4. Combine grouping with appropriate mode for insights
  5. Adjust threshold if default filtering is too aggressive

✓ TYPICAL WORKFLOWS:
  • Exploratory Analysis: Use completeness mode + grouping
  • Detailed Investigation: Use binary mode + specific columns
  • Quality Reports: Compare filtered vs unfiltered views
  • Pattern Discovery: Try different grouping variables
""")

print("="*80)
print("Demo completed! All features demonstrated with visual comparisons.")
print("="*80)
