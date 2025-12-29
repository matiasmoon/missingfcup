"""
Demo: Simple Introduction to missingfcup

A beginner-friendly demonstration of missingfcup's core features using a small dataset.
Perfect for learning the basics before tackling larger, real-world datasets.

Features demonstrated:
- Basic heatmap and barchart creation
- Grouping and sorting
- Binary vs Completeness modes (NEW!)
- High missingness filtering (NEW!)
- Side-by-side panel comparisons
"""

from missingfcup import MissingObject, Panel
import pandas as pd

print("="*80)
print("SIMPLE DEMO - INTRODUCTION TO MISSINGFCUP")
print("="*80)

# ============================================================================
# CREATE SAMPLE DATASET
# ============================================================================
print("\nCreating sample dataset with missing values...")

df = pd.DataFrame({
    "ID": [1, 2, 3, 4, 5, 6, 7, 8],
    "Name": ["Alice", "Bob", "Charlie", None, "Eve", "Frank", None, "Hannah"],
    "Age": [25, None, 35, 28, None, 42, 31, 29],
    "City": ["NYC", "LA", None, "NYC", "LA", "NYC", "LA", "NYC"],
    "Salary": [50000, 60000, None, 45000, None, 70000, 48000, None],
    "Department": ["Sales", "Eng", "Sales", "HR", "Eng", "Sales", "HR", "Eng"],
    "Bonus": [None, 5000, None, None, 3000, 8000, None, 4000],
    "Rarely_Filled": [None, None, None, None, None, None, None, 100],  # 87.5% missing
})

print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nDataset preview:\n{df}")

missing = MissingObject(df)

# ============================================================================
# SECTION 1: BASIC VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: BASIC VISUALIZATIONS")
print("="*80)

# Example 1.1: Basic heatmap
print("\n1.1 Basic Heatmap - See Missing Value Patterns")

heatmap = missing.heatmap(
    title="Simple Dataset: Missing Value Heatmap",
    show_colorscale_legend=True,
    figure_size_pixels=(900, 600)
)
heatmap.show()

# Example 1.2: Bar chart
print("\n1.2 Bar Chart - Column Completeness Overview")

barchart = missing.barchart()
barchart.show()

# ============================================================================
# SECTION 2: HIGH MISSINGNESS FILTERING (NEW FEATURE!)
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: HIGH MISSINGNESS FILTERING")
print("="*80)
print("NEW FEATURE: Automatically exclude columns with too much missing data")

# Example 2.1: Comparison - Filtered vs Unfiltered
print("\n2.1 Comparing: Default Filtering vs No Filtering")

panel_filter = Panel([
    missing.heatmap(
        ignore_high_missingness=True,  # Default: exclude columns with ≥90% missing
        title="Filtered (≥90% missing excluded)",
        show_colorscale_legend=True
    ),
    missing.heatmap(
        ignore_high_missingness=False,  # Include all columns
        title="Unfiltered (All columns)",
        show_colorscale_legend=True
    ),
])

panel_filter.show()

# ============================================================================
# SECTION 3: GROUPING AND SORTING
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: GROUPING AND SORTING")
print("="*80)

# Example 3.1: Group by Department
print("\n3.1 Grouping by Department")

heatmap_dept = missing.heatmap(
    group_by="Department",
    group_categories=["Sales", "HR", "Eng"],  # Custom order
    title="Grouped by Department",
    show_colorscale_legend=True,
    figure_size_pixels=(900, 600)
)
heatmap_dept.show()

# Example 3.2: Group by Age (numeric)
print("\n3.2 Grouping by Age (Ascending)")

heatmap_age = missing.heatmap(
    group_by="Age",
    group_direction="ascending",
    title="Grouped by Age (Ascending)",
    show_colorscale_legend=True,
    figure_size_pixels=(900, 600)
)
heatmap_age.show()

# ============================================================================
# SECTION 4: BINARY VS COMPLETENESS MODE (NEW FEATURE!)
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: BINARY vs COMPLETENESS VISUALIZATION")
print("="*80)
print("NEW FEATURE: Choose how to visualize missing data")

# Example 4.1: Side-by-side comparison
print("\n4.1 Comparing: Binary Mode vs Completeness Mode")

panel_modes = Panel([
    missing.heatmap(
        group_by="Department",
        group_categories=["Sales", "HR", "Eng"],
        group_by_mode="binary",  # Default: show individual cells
        title="Binary Mode (Cell-Level)",
        show_colorscale_legend=True
    ),
    missing.heatmap(
        group_by="Department",
        group_categories=["Sales", "HR", "Eng"],
        group_by_mode="completeness",  # NEW: show row completeness %
        title="Completeness Mode (Row-Level %)",
        show_colorscale_legend=True
    ),
])

panel_modes.show()

# ============================================================================
# SECTION 5: MULTI-PANEL COMPARISONS
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: MULTI-PANEL COMPARISONS")
print("="*80)

# Example 5.1: Four-panel comprehensive view
print("\n5.1 Four Different Configurations Side-by-Side")

panel = Panel([
    missing.heatmap(
        title="Default View",
        show_colorscale_legend=True
    ),
    missing.heatmap(
        group_by="Department",
        group_categories=["Sales", "HR", "Eng"],
        title="Grouped by Department",
        show_colorscale_legend=True
    ),
    missing.heatmap(
        group_by="Department",
        group_categories=["Sales", "HR", "Eng"],
        group_by_mode="completeness",
        title="Completeness by Department",
        show_colorscale_legend=True
    ),
    missing.heatmap(
        ignore_high_missingness=False,
        title="All Columns (Unfiltered)",
        show_colorscale_legend=True
    ),
])

panel.show()

# ============================================================================
# SECTION 6: COLUMN SELECTION
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: FOCUSING ON SPECIFIC COLUMNS")
print("="*80)

# Example 6.1: Select specific columns
print("\n6.1 Analyzing Key Columns Only")

key_cols = ["Name", "Age", "City", "Salary"]
heatmap_selected = missing.heatmap(
    selected_columns=key_cols,
    group_by="City",
    title="Key Columns Analysis",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 500)
)
heatmap_selected.show()

# ============================================================================
# SECTION 7: CUSTOM STYLING
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: CUSTOM COLOR SCHEMES")
print("="*80)

# Example 7.1: Custom colors
print("\n7.1 Custom Color Scheme")

heatmap_custom = missing.heatmap(
    group_by="Department",
    group_categories=["Sales", "HR", "Eng"],
    present_color="#1f77b4",  # Blue
    missing_color="#ff7f0e",  # Orange
    title="Custom Colors: Blue/Orange",
    show_colorscale_legend=True,
    figure_size_pixels=(900, 600)
)
heatmap_custom.show()

# ============================================================================
# SECTION 8: COMPLETE COMPARISON GRID
# ============================================================================
print("\n" + "="*80)
print("SECTION 8: COMPREHENSIVE 4-PANEL COMPARISON")
print("="*80)

# Example 8.1: 2x2 grid showing all feature combinations
print("\n8.1 All Features Combined: 2×2 Grid")

panel_comprehensive = Panel([
    missing.heatmap(
        group_by="Department",
        group_categories=["Sales", "HR", "Eng"],
        group_by_mode="binary",
        title="Binary Mode (Filtered)",
        show_colorscale_legend=True
    ),
    missing.heatmap(
        group_by="Department",
        group_categories=["Sales", "HR", "Eng"],
        group_by_mode="completeness",
        title="Completeness Mode (Filtered)",
        show_colorscale_legend=True
    ),
    missing.heatmap(
        group_by="Department",
        group_categories=["Sales", "HR", "Eng"],
        group_by_mode="binary",
        ignore_high_missingness=False,
        title="Binary Mode (All Columns)",
        show_colorscale_legend=True
    ),
    missing.heatmap(
        group_by="Department",
        group_categories=["Sales", "HR", "Eng"],
        group_by_mode="completeness",
        ignore_high_missingness=False,
        title="Completeness Mode (All Columns)",
        show_colorscale_legend=True
    ),
])

panel_comprehensive.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY - WHAT YOU LEARNED")
print("="*80)
print("""
✅ BASIC FEATURES:
   • Heatmap: Visual grid of missing/present values
   • Barchart: Column completeness overview
   • Group by: Sort rows by column values
   • Custom colors: Personalize your visualizations

✅ NEW FEATURES:
   1. High Missingness Filtering
      • ignore_high_missingness=True (default)
      • Excludes columns with ≥90% missing
      • Keeps analysis focused on meaningful data

   2. Visualization Modes
      • Binary mode (default): See individual cells
      • Completeness mode: See row-level quality %
      • Use completeness for group comparisons

✅ WHEN TO USE WHAT:
   📊 Binary Mode:
      → Finding specific missing values
      → Data quality audits
      → Detailed investigations

   📈 Completeness Mode:
      → Comparing groups
      → Overall quality assessment
      → Pattern recognition

✅ BEST PRACTICES:
   1. Start with default filtering
   2. Use grouping to reveal patterns
   3. Compare binary vs completeness modes
   4. Create multi-panel views for insights
   5. Customize colors for presentations

Next Steps:
• Try with your own data!
• Check out demo_collisions_dataset.py for real-world examples
• Explore demo_groupby.py for advanced grouping techniques
""")

print("="*80)
print("Demo completed! You're ready to analyze your own data.")
print("="*80)
