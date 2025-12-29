"""
Demo: Categorical Data Analysis with missingfcup

Demonstrates how to analyze missing data patterns in datasets with categorical variables.
Shows best practices for grouping and visualizing categorical data.

Features demonstrated:
- Working with mixed numeric and categorical data
- Custom category ordering
- Binary vs Completeness modes for categories (NEW!)
- High missingness filtering (NEW!)
- Multi-panel categorical comparisons
"""

from missingfcup import MissingObject, Panel
import pandas as pd
from plotly.subplots import make_subplots

print("="*80)
print("CATEGORICAL DATA DEMO - WORKING WITH MIXED DATA TYPES")
print("="*80)

# ============================================================================
# CREATE SAMPLE DATASET WITH CATEGORICAL DATA
# ============================================================================
print("\nCreating sample dataset with categorical and numeric columns...")

df = pd.DataFrame({
    "Product_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "Category": ["Electronics", "Clothing", "Electronics", "Food", "Clothing",
                 "Food", "Electronics", "Clothing", "Food", "Electronics"],
    "Brand": ["Apple", "Nike", None, "Nestle", "Adidas", None, "Samsung", None, "Kraft", "Sony"],
    "Price": [999, None, 1299, 15, 80, None, 899, 65, None, 749],
    "Stock": [50, 120, None, 200, None, 150, 75, None, 180, 90],
    "Rating": ["Excellent", "Good", None, "Fair", "Excellent", None, "Good", "Excellent", "Fair", None],
    "Supplier": ["SupplierA", None, "SupplierA", "SupplierB", None, "SupplierB", "SupplierA", None, "SupplierB", "SupplierA"],
    "Rarely_Available": [None, None, None, None, None, None, None, None, None, "Limited"],  # 90% missing
})

print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nDataset preview:\n{df}")

missing = MissingObject(df)

# ============================================================================
# SECTION 1: BASIC CATEGORICAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: BASIC CATEGORICAL ANALYSIS")
print("="*80)

# Example 1.1: Basic heatmap
print("\n1.1 Basic Heatmap - Mixed Data Types")

heatmap_basic = missing.heatmap(
    title="Product Data: Missing Value Patterns",
    show_colorscale_legend=True,
    figure_size_pixels=(1000, 600)
)
heatmap_basic.show()

# Example 1.2: Bar chart showing completeness
print("\n1.2 Bar Chart - Column Completeness")

barchart = missing.barchart()
barchart.show()

# ============================================================================
# SECTION 2: GROUPING BY CATEGORY
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: GROUPING BY CATEGORICAL VARIABLES")
print("="*80)

# Example 2.1: Group by Product Category
print("\n2.1 Grouping by Product Category")

heatmap_cat = missing.heatmap(
    group_by="Category",
    group_categories=["Electronics", "Clothing", "Food"],  # Custom order
    title="Missing Patterns by Product Category",
    show_colorscale_legend=True,
    figure_size_pixels=(1000, 600)
)
heatmap_cat.show()

# Example 2.2: Group by Rating (ordinal categorical)
print("\n2.2 Grouping by Rating (Ordinal)")

heatmap_rating = missing.heatmap(
    group_by="Rating",
    group_categories=["Fair", "Good", "Excellent"],  # Ordinal order
    title="Missing Patterns by Rating Level",
    show_colorscale_legend=True,
    figure_size_pixels=(1000, 600)
)
heatmap_rating.show()

# ============================================================================
# SECTION 3: BINARY VS COMPLETENESS FOR CATEGORIES (NEW!)
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: BINARY vs COMPLETENESS FOR CATEGORICAL GROUPS")
print("="*80)
print("NEW FEATURE: Compare data quality across categorical groups")

# Example 3.1: Binary mode - see specific missing cells per category
print("\n3.1 Binary Mode: Cell-Level View by Category")

heatmap_binary = missing.heatmap(
    group_by="Category",
    group_categories=["Electronics", "Clothing", "Food"],
    group_by_mode="binary",  # Default: individual cells
    title="Binary Mode: Missing Cells by Category",
    show_colorscale_legend=True,
    figure_size_pixels=(900, 600)
)
heatmap_binary.show()

# Example 3.2: Completeness mode - overall quality per category
print("\n3.2 Completeness Mode: Overall Quality by Category")
print("    NEW: Shows which categories have better data completeness")

heatmap_comp = missing.heatmap(
    group_by="Category",
    group_categories=["Electronics", "Clothing", "Food"],
    group_by_mode="completeness",  # NEW: row completeness %
    title="Completeness Mode: Data Quality by Category",
    show_colorscale_legend=True,
    figure_size_pixels=(900, 600)
)
heatmap_comp.show()

# Example 3.3: Side-by-side comparison
print("\n3.3 Comparing: Binary vs Completeness for Categories")

fig_compare = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Binary: Individual Patterns", "Completeness: Category Quality"),
    horizontal_spacing=0.15
)

for trace in heatmap_binary.fig.data:
    fig_compare.add_trace(trace, row=1, col=1)
for trace in heatmap_comp.fig.data:
    fig_compare.add_trace(trace, row=1, col=2)

fig_compare.update_layout(
    title_text="Categorical Analysis: Binary vs Completeness Modes",
    height=700,
    showlegend=False
)
fig_compare.show()

# ============================================================================
# SECTION 4: HIGH MISSINGNESS FILTERING WITH CATEGORIES
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: HIGH MISSINGNESS FILTERING")
print("="*80)
print("NEW FEATURE: Filter out columns with too much missing data")

# Example 4.1: Comparison
print("\n4.1 Comparing: Filtered vs Unfiltered with Categorical Grouping")

heatmap_filtered = missing.heatmap(
    group_by="Category",
    group_categories=["Electronics", "Clothing", "Food"],
    group_by_mode="completeness",
    ignore_high_missingness=True,  # Default: exclude ≥90% missing
    title="Filtered (≥90% missing excluded)",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 600)
)

heatmap_unfiltered = missing.heatmap(
    group_by="Category",
    group_categories=["Electronics", "Clothing", "Food"],
    group_by_mode="completeness",
    ignore_high_missingness=False,  # Include all columns
    title="Unfiltered (All columns)",
    show_colorscale_legend=True,
    figure_size_pixels=(800, 600)
)

fig_filter = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Filtered View", "Full View"),
    horizontal_spacing=0.15
)

for trace in heatmap_filtered.fig.data:
    fig_filter.add_trace(trace, row=1, col=1)
for trace in heatmap_unfiltered.fig.data:
    fig_filter.add_trace(trace, row=1, col=2)

fig_filter.update_layout(
    title_text="Impact of Filtering on Categorical Analysis",
    height=700,
    showlegend=False
)
fig_filter.show()

# ============================================================================
# SECTION 5: COMPARING DIFFERENT CATEGORICAL GROUPINGS
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: COMPARING DIFFERENT CATEGORICAL GROUPINGS")
print("="*80)

# Example 5.1: Three different groupings side-by-side
print("\n5.1 Three-Panel Comparison: Different Category Variables")

heatmap_by_cat = missing.heatmap(
    group_by="Category",
    group_categories=["Electronics", "Clothing", "Food"],
    group_by_mode="completeness",
    title="By Product Category",
    show_colorscale_legend=True,
    figure_size_pixels=(600, 550)
)

heatmap_by_rating = missing.heatmap(
    group_by="Rating",
    group_categories=["Fair", "Good", "Excellent"],
    group_by_mode="completeness",
    title="By Rating Level",
    show_colorscale_legend=True,
    figure_size_pixels=(600, 550)
)

heatmap_by_supplier = missing.heatmap(
    group_by="Supplier",
    group_by_mode="completeness",
    title="By Supplier",
    show_colorscale_legend=True,
    figure_size_pixels=(600, 550)
)

fig_multi = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Product Category", "Rating Level", "Supplier"),
    horizontal_spacing=0.15
)

for trace in heatmap_by_cat.fig.data:
    fig_multi.add_trace(trace, row=1, col=1)
for trace in heatmap_by_rating.fig.data:
    fig_multi.add_trace(trace, row=1, col=2)
for trace in heatmap_by_supplier.fig.data:
    fig_multi.add_trace(trace, row=1, col=3)

fig_multi.update_layout(
    title_text="Completeness Across Different Categorical Variables",
    height=700,
    width=1600,
    showlegend=False
)
fig_multi.show()

# ============================================================================
# SECTION 6: MULTI-PANEL USING PANEL CLASS
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: MULTI-PANEL CATEGORICAL VIEWS")
print("="*80)

# Example 6.1: Four-panel categorical analysis
print("\n6.1 Four-Panel Categorical Analysis")

panel = Panel([
    missing.heatmap(
        group_by="Category",
        group_categories=["Electronics", "Clothing", "Food"],
        group_by_mode="binary",
        title="Category (Binary)",
        show_colorscale_legend=True
    ),
    missing.heatmap(
        group_by="Category",
        group_categories=["Electronics", "Clothing", "Food"],
        group_by_mode="completeness",
        title="Category (Completeness)",
        show_colorscale_legend=True
    ),
    missing.heatmap(
        group_by="Rating",
        group_categories=["Fair", "Good", "Excellent"],
        group_by_mode="binary",
        title="Rating (Binary)",
        show_colorscale_legend=True
    ),
    missing.heatmap(
        group_by="Rating",
        group_categories=["Fair", "Good", "Excellent"],
        group_by_mode="completeness",
        title="Rating (Completeness)",
        show_colorscale_legend=True
    ),
])

panel.show()

# ============================================================================
# SECTION 7: CUSTOM CATEGORY ORDERING
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: IMPORTANCE OF CATEGORY ORDERING")
print("="*80)

# Example 7.1: Different ordering strategies
print("\n7.1 Comparing: Different Category Orders")

# Alphabetical order
heatmap_alpha = missing.heatmap(
    group_by="Category",
    # No group_categories = pandas default (alphabetical)
    group_by_mode="completeness",
    title="Alphabetical Order",
    show_colorscale_legend=True,
    figure_size_pixels=(700, 550)
)

# Custom logical order
heatmap_logical = missing.heatmap(
    group_by="Category",
    group_categories=["Electronics", "Clothing", "Food"],  # Logical order
    group_by_mode="completeness",
    title="Logical Order (Electronics→Clothing→Food)",
    show_colorscale_legend=True,
    figure_size_pixels=(700, 550)
)

fig_order = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Default (Alphabetical)", "Custom Logical Order"),
    horizontal_spacing=0.15
)

for trace in heatmap_alpha.fig.data:
    fig_order.add_trace(trace, row=1, col=1)
for trace in heatmap_logical.fig.data:
    fig_order.add_trace(trace, row=1, col=2)

fig_order.update_layout(
    title_text="Impact of Category Ordering on Pattern Visibility",
    height=700,
    showlegend=False
)
fig_order.show()

# ============================================================================
# SECTION 8: COMPREHENSIVE CATEGORICAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 8: COMPREHENSIVE 4-PANEL CATEGORICAL ANALYSIS")
print("="*80)

# Example 8.1: 2x2 grid showing all combinations
print("\n8.1 All Features Combined for Categorical Data")

heatmap_1 = missing.heatmap(
    group_by="Category",
    group_categories=["Electronics", "Clothing", "Food"],
    group_by_mode="binary",
    title="Binary + Filtered",
    show_colorscale_legend=True,
    figure_size_pixels=(500, 450)
)

heatmap_2 = missing.heatmap(
    group_by="Category",
    group_categories=["Electronics", "Clothing", "Food"],
    group_by_mode="completeness",
    title="Completeness + Filtered",
    show_colorscale_legend=True,
    figure_size_pixels=(500, 450)
)

heatmap_3 = missing.heatmap(
    group_by="Category",
    group_categories=["Electronics", "Clothing", "Food"],
    group_by_mode="binary",
    ignore_high_missingness=False,
    title="Binary + All Columns",
    show_colorscale_legend=True,
    figure_size_pixels=(500, 450)
)

heatmap_4 = missing.heatmap(
    group_by="Category",
    group_categories=["Electronics", "Clothing", "Food"],
    group_by_mode="completeness",
    ignore_high_missingness=False,
    title="Completeness + All Columns",
    show_colorscale_legend=True,
    figure_size_pixels=(500, 450)
)

fig_final = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Binary Mode (Filtered)",
        "Completeness Mode (Filtered)",
        "Binary Mode (All Columns)",
        "Completeness Mode (All Columns)"
    ),
    horizontal_spacing=0.15,
    vertical_spacing=0.2
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
    title_text="Comprehensive Categorical Analysis: All Feature Combinations",
    height=1000,
    width=1200,
    showlegend=False
)
fig_final.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY - WORKING WITH CATEGORICAL DATA")
print("="*80)
print("""
✅ KEY INSIGHTS FOR CATEGORICAL DATA:

1. GROUPING STRATEGIES:
   • Use categorical columns to reveal group-specific patterns
   • Custom category ordering makes patterns more visible
   • Compare multiple categorical variables

2. CATEGORY ORDERING:
   • Alphabetical (default): Simple, consistent
   • Logical: Groups related categories together
   • Ordinal: For ranked categories (Fair→Good→Excellent)
   • Custom: Based on business logic or frequency

3. VISUALIZATION MODES (NEW!):
   📊 Binary Mode:
      → See which specific fields are missing per category
      → Identify category-specific data collection issues
      → Useful for quality audits

   📈 Completeness Mode:
      → Compare overall data quality across categories
      → Identify which categories have best/worst data
      → Perfect for reports and dashboards

4. HIGH MISSINGNESS FILTERING (NEW!):
   • Automatically excludes columns with ≥90% missing
   • Keeps analysis focused on meaningful categorical data
   • Customizable threshold for your needs

✅ BEST PRACTICES FOR CATEGORICAL DATA:

1. Start with completeness mode to identify problematic categories
2. Use binary mode to investigate specific issues
3. Define custom category orders for better pattern visibility
4. Compare multiple categorical variables side-by-side
5. Filter high-missingness columns for cleaner analysis

✅ COMMON USE CASES:

📦 Product Analysis:
   → Group by category/brand to find data gaps
   → Use completeness mode to prioritize data collection
   → Identify categories needing more information

👥 Customer Segmentation:
   → Group by demographics to see coverage
   → Compare data quality across segments
   → Find which segments have incomplete profiles

📍 Geographic Analysis:
   → Group by location/region/country
   → Identify areas with poor data collection
   → Prioritize data quality efforts geographically

Next Steps:
• Apply these techniques to your categorical data
• Try different category orderings to reveal patterns
• Compare binary vs completeness for your use case
• Explore demo_groupby.py for advanced grouping techniques
""")

print("="*80)
print("Demo completed! You're ready to analyze categorical data.")
print("="*80)
