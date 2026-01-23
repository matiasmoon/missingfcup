"""
Demo: Panel Comparisons with missingfcup - NYC Collisions Dataset

Demonstrates how to use the Panel class to create side-by-side comparisons
of different missing data visualizations using real-world data.
Perfect for comparing different views, groupings, and modes in a single display.

Features demonstrated:
- Creating multi-panel layouts with real data
- Comparing different grouping strategies (Borough, Contributing Factor)
- Binary vs Completeness mode comparisons
- Filtered vs Unfiltered comparisons
- Different sorting and visualization options
"""

from missingfcup import MissingObject, Panel
import pandas as pd

print("="*80)
print("PANEL DEMO - MULTI-PANEL COMPARISONS WITH NYC COLLISIONS DATA")
print("="*80)

# ============================================================================
# LOAD DATASET
# ============================================================================
print("\nLoading NYC Motor Vehicle Collisions dataset from GitHub...")

try:
    collisions = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {collisions.shape[0]:,} rows × {collisions.shape[1]} columns")
    print(f"  Columns: {list(collisions.columns)}")

    # Show a preview of the data
    print(f"\nDataset preview:")
    print(collisions.head())

except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    print("  Please check your internet connection and try again.")
    exit(1)

missing = MissingObject(collisions)

# ============================================================================
# SECTION 1: BASIC PANEL USAGE
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: BASIC PANEL USAGE")
print("="*80)

# Example 1.1: Two-panel comparison
print("\n1.1 Two-Panel Comparison: Heatmap vs Bar Chart")
print("    Panel automatically arranges plots in a grid (max 2 columns)")

panel_basic = Panel([
    missing.heatmap(
        title="Heatmap: Missing Value Patterns in Collisions",
        show_colorscale_legend=True,
        figure_size_pixels=(500, 900)
    )
])

panel_basic.show()

# ============================================================================
# SECTION 2: COMPARING FILTER SETTINGS
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: COMPARING FILTER SETTINGS")
print("="*80)

# Example 2.1: Filtered vs Unfiltered
print("\n2.1 Side-by-Side: Filtered vs Unfiltered Views")
print("    See the impact of high missingness filtering")

panel_filter = Panel([
    missing.heatmap(
        ignore_high_missingness=True,  # Default: exclude ≥90% missing
        title="Filtered (≥90% missing excluded)",
        show_colorscale_legend=True,
        figure_size_pixels=(1000, 500)
    ),
    missing.heatmap(
        ignore_high_missingness=False,  # Include all columns
        title="Unfiltered (All columns)",
        show_colorscale_legend=True,
        figure_size_pixels=(1000, 500)
    ),
])

panel_filter.show()

# ============================================================================
# SECTION 3: COMPARING GROUPING STRATEGIES
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: COMPARING GROUPING STRATEGIES")
print("="*80)

# Example 3.1: Different grouping columns
print("\n3.1 Two Different Grouping Variables")
print("    Compare how different columns reveal different patterns")
print("    Borough shows geographic patterns, Contributing Factor shows causal patterns")

panel_groupings = Panel([
    missing.heatmap(
        sort_by_columns="BOROUGH",
        group_by_mode="binary",
        title="Grouped by Borough",
        show_colorscale_legend=True,
        figure_size_pixels=(1000, 500)
    ),
    missing.heatmap(
        sort_by_columns="CONTRIBUTING FACTOR VEHICLE 1",
        group_by_mode="binary",
        title="Grouped by Contributing Factor",
        show_colorscale_legend=True,
        max_columns_by_completeness=10,  # Limit columns for readability
        figure_size_pixels=(1000, 500)
    ),
])

panel_groupings.show()

# ============================================================================
# SECTION 4: BINARY VS COMPLETENESS MODES
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: BINARY VS COMPLETENESS MODES")
print("="*80)

# Example 4.1: Same grouping, different modes
print("\n4.1 Binary vs Completeness for Borough Grouping")
print("    Binary mode shows cell-level detail, Completeness shows % missing")
print("    Helps identify which boroughs have better data quality")

panel_modes = Panel([
    missing.heatmap(
        sort_by_columns="BOROUGH",
        group_by_mode="binary",
        title="Binary Mode: Cell-Level Detail by Borough",
        show_colorscale_legend=True,
        figure_size_pixels=(1000, 500)
    ),
    missing.heatmap(
        sort_by_columns="BOROUGH",
        group_by_mode="completeness",
        title="Completeness Mode: % Missing by Borough",
        show_colorscale_legend=True,
        figure_size_pixels=(1000, 500)
    ),
])

panel_modes.show()

# ============================================================================
# SECTION 5: FOUR-PANEL COMPREHENSIVE VIEW
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: FOUR-PANEL COMPREHENSIVE VIEW")
print("="*80)

# Example 5.1: 2x2 grid with all combinations
print("\n5.1 Four-Panel Analysis: All Feature Combinations")
print("    Binary/Completeness × Filtered/Unfiltered for Borough Analysis")
print("    Complete view of data quality across all NYC boroughs")

panel_comprehensive = Panel([
    missing.heatmap(
        sort_by_columns="BOROUGH",
        group_by_mode="binary",
        ignore_high_missingness=True,
        title="Binary + Filtered",
        show_colorscale_legend=True,
        figure_size_pixels=(450, 400)
    ),
    missing.heatmap(
        sort_by_columns="BOROUGH",
        group_by_mode="completeness",
        ignore_high_missingness=True,
        title="Completeness + Filtered",
        show_colorscale_legend=True,
        figure_size_pixels=(450, 400)
    ),
    missing.heatmap(
        sort_by_columns="BOROUGH",
        group_by_mode="binary",
        ignore_high_missingness=False,
        title="Binary + Unfiltered",
        show_colorscale_legend=True,
        figure_size_pixels=(450, 400)
    ),
    missing.heatmap(
        sort_by_columns="BOROUGH",
        group_by_mode="completeness",
        ignore_high_missingness=False,
        title="Completeness + Unfiltered",
        show_colorscale_legend=True,
        figure_size_pixels=(450, 400)
    ),
])

panel_comprehensive.show()

# ============================================================================
# SECTION 6: COLUMN SELECTION COMPARISON
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: COLUMN SELECTION COMPARISON")
print("="*80)

# Example 6.1: Most vs Least complete columns
print("\n6.1 Most Complete vs Least Complete Columns")
print("    Focus analysis on different aspects of data quality")

panel_completeness = Panel([
    missing.heatmap(
        sort_by_columns="BOROUGH",
        completeness_mode="most",
        max_columns_by_completeness=10,
        group_by_mode="completeness",
        title="10 Most Complete Columns by Borough",
        show_colorscale_legend=True,
        figure_size_pixels=(600, 500)
    ),
    missing.heatmap(
        sort_by_columns="BOROUGH",
        completeness_mode="least",
        max_columns_by_completeness=10,
        group_by_mode="completeness",
        title="10 Least Complete Columns by Borough",
        show_colorscale_legend=True,
        figure_size_pixels=(600, 500)
    ),
])

panel_completeness.show()

# ============================================================================
# SECTION 7: SAVING PANELS
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: SAVING PANELS")
print("="*80)

# Example 7.1: Save panel to HTML file
print("\n7.1 Saving Panel to HTML File")
print("    Panels can be saved just like individual plots")
print("    Perfect for reports, presentations, or sharing analysis")

output_path = "/tmp/nyc_collisions_panel_comparison.html"
panel_modes.save(output_path)
print(f"✓ Panel saved to: {output_path}")
print("  Open this file in a browser to view the interactive visualization")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY - PANEL CLASS WITH REAL-WORLD DATA")
print("="*80)
print("""
✅ KEY FEATURES OF THE PANEL CLASS:

1. AUTOMATIC LAYOUT:
   • Panels automatically arrange plots in a grid
   • Maximum 2 columns for readability
   • Rows adjust based on number of plots

2. EASY COMPARISONS:
   • Side-by-side views of different settings
   • Compare groupings, modes, filters, etc.
   • See differences at a glance

3. UNIFIED DISPLAY:
   • All plots in one figure
   • Consistent sizing and styling
   • Single .show() or .save() call

✅ WHAT WE LEARNED FROM NYC COLLISIONS DATA:

🗺️ Geographic Patterns:
   → Borough-level analysis reveals data quality differences
   → Some boroughs have more complete reporting than others
   → Panel comparisons make geographic disparities obvious

🚗 Contributing Factors:
   → Different factors have different levels of missing data
   → Grouping by contributing factor reveals reporting biases
   → Binary vs completeness modes show different insights

📊 Filter Impact:
   → High missingness filtering cleans up analysis
   → Side-by-side comparison shows what's excluded
   → Helps focus on actionable columns

🎯 Mode Selection:
   → Binary mode: Best for seeing individual patterns
   → Completeness mode: Best for overall quality assessment
   → Panel lets you see both perspectives simultaneously

✅ COMMON USE CASES FOR PANELS:

📈 Data Quality Reports:
   → Multiple views of same dataset
   → Show stakeholders different aspects
   → Highlight problem areas

🔍 Exploratory Analysis:
   → Compare different grouping variables
   → Try different visualization modes
   → Identify meaningful patterns quickly

📋 Documentation:
   → Save panels as HTML for reports
   → Include in presentations
   → Share interactive visualizations

🎓 Research & Analysis:
   → Systematic comparison of approaches
   → Support for methodology decisions
   → Clear visualization of trade-offs

✅ BEST PRACTICES FOR PANELS:

1. Use 2-4 plots per panel for clarity
2. Keep titles concise and descriptive
3. Use consistent figure sizes within a panel
4. Compare related views (same data, different settings)
5. Save panels for reports and documentation

✅ PANEL WORKFLOW:

1. Load your data and create MissingObject
2. Create multiple heatmaps with different settings
3. Pass them as a list to Panel([plot1, plot2, ...])
4. Call .show() to display or .save(path) to export
5. Plots are automatically arranged in a clean grid

Next Steps:
• Apply Panel to your own datasets
• Experiment with different grouping variables
• Create comprehensive 4-panel reports
• Save panels for presentations and documentation
• Explore demo_collisions_dataset.py for more collisions analysis
""")

print("="*80)
print("Demo completed! You're ready to use Panels with real data.")
print("="*80)
