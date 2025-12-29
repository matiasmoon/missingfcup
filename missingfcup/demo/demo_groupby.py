"""
Demo: Grouping rows in missingness heatmaps

This demo shows how to use the group_by parameter to reveal patterns
in missing data by grouping/sorting rows according to specific columns.
"""

from missingfcup import MissingObject
import pandas as pd

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

# Example 1: Basic heatmap (no grouping)
print("Example 1: Basic heatmap (Y-axis shows first column values)")
print("  Y-axis will display 'age' values in original order")
heatmap1 = missing.heatmap(
    title="Default - Y-axis shows 'age' column",
    show_colorscale_legend=True
)
heatmap1.show()

# Example 2: Group by numeric column (ascending)
print("Example 2: Group by age (ascending)")
heatmap2 = missing.heatmap(
    group_by="age",
    group_direction="ascending",
    title="Ordered by Age (Ascending)",
    show_colorscale_legend=True
)
heatmap2.show()

# Example 3: Group by numeric column (descending)
print("Example 3: Group by age (descending)")
heatmap3 = missing.heatmap(
    group_by="age",
    group_direction="descending",
    title="Ordered by Age (Descending)",
    show_colorscale_legend=True
)
heatmap3.show()

# Example 4: Group by categorical column with custom categories
print("Example 4: Group by education level (custom category order)")
# Calculate appropriate height: ~30 pixels per row + margins
num_rows = len(df)
appropriate_height = max(300, num_rows * 30 + 100)  # min 300px, 30px per row + 100px margins
heatmap4 = missing.heatmap(
    group_by="education",
    group_categories=["Low", "Med", "High"],  # Custom order
    title="Ordered by Education (Low → Med → High)",
    show_colorscale_legend=True,
    figure_size_pixels=(900, appropriate_height)
)
heatmap4.show()

# Example 5: Group by two columns
print("Example 5: Group by city, then age")
heatmap5 = missing.heatmap(
    group_by=["city", "age"],
    group_direction=["ascending", "ascending"],
    title="Ordered by City, then Age",
    show_colorscale_legend=True,
    figure_size_pixels=(1000, 700)
)
heatmap5.show()

# Example 6: Mixed ordering - categorical + numeric
print("Example 6: Group by education (custom), then age (descending)")
heatmap6 = missing.heatmap(
    group_by=["education", "age"],
    group_categories=[["Low", "Med", "High"], None],  # Only first column has categories
    group_direction=["ascending", "descending"],
    title="Mixed Ordering: Education (Low→High), Age (High→Low)",
    show_colorscale_legend=True,
    figure_size_pixels=(1000, 700)
)
heatmap6.show()

print("\n" + "="*60)
print("Summary of Y-Axis Behavior:")
print("="*60)
print("• Default (no group_by): Y-axis shows first column ('age') values")
print("• With group_by: Y-axis shows the grouping column(s) values")
print("• Grouping columns are moved to the left and highlighted")
print("• Orange borders mark the grouping columns")
print("• Missing values in grouping columns appear as 'nan'")
print("="*60)

# Example 7: Group by with completeness mode (NEW FEATURE!)
print("\nExample 7: Group by with COMPLETENESS visualization")
print("  Instead of binary present/missing, show row completeness percentage")
heatmap7 = missing.heatmap(
    group_by="education",
    group_categories=["Low", "Med", "High"],
    group_by_mode="completeness",  # NEW: Show completeness percentage per row
    title="Completeness by Education Level",
    show_colorscale_legend=True,
    figure_size_pixels=(900, appropriate_height)
)
heatmap7.show()

# Example 8: Demonstrate ignore_high_missingness parameter
print("\nExample 8: Controlling high missingness filtering")

# Create a dataset with some columns that have very high missingness
df_with_sparse = df.copy()
df_with_sparse['rarely_filled_1'] = [None] * 9 + [100]  # 90% missing
df_with_sparse['rarely_filled_2'] = [None] * 9 + [200]  # 90% missing
df_with_sparse['almost_empty'] = [None] * 10  # 100% missing

missing_sparse = MissingObject(df_with_sparse)

print("  8a. Default behavior (ignore columns with ≥90% missing)")
heatmap8a = missing_sparse.heatmap(
    title="Default - Columns with ≥90% missing are excluded",
    show_colorscale_legend=True
)
heatmap8a.show()

print("  8b. Include ALL columns (ignore_high_missingness=False)")
heatmap8b = missing_sparse.heatmap(
    ignore_high_missingness=False,  # NEW: Include all columns
    title="All Columns Included (even 90%+ missing)",
    show_colorscale_legend=True
)
heatmap8b.show()

print("  8c. Custom threshold (exclude columns with ≥80% missing)")
heatmap8c = missing_sparse.heatmap(
    high_missingness_threshold=0.8,  # NEW: Custom threshold
    title="Exclude Columns with ≥80% Missing",
    show_colorscale_legend=True
)
heatmap8c.show()