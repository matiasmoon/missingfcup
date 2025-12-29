"""
Demo: Ordering rows in missingness heatmaps

This demo shows how to use the order_by parameter to reveal patterns
in missing data by ordering rows according to specific columns.
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
    'bonus': [5000, None, None, 8000, None, 6000, None, None, 7000, None],
})

missing = MissingObject(df)

# Example 1: Basic heatmap (no ordering)
print("Example 1: Basic heatmap (rows in original order)")
heatmap1 = missing.heatmap(
    title="No Ordering - Original Row Order",
    show_colorscale_legend=True
)
heatmap1.show()

# Example 2: Order by numeric column (ascending)
print("Example 2: Order by age (ascending)")
heatmap2 = missing.heatmap(
    order_by="age",
    order_direction="ascending",
    title="Ordered by Age (Ascending)",
    show_colorscale_legend=True
)
heatmap2.show()

# Example 3: Order by numeric column (descending)
print("Example 3: Order by age (descending)")
heatmap3 = missing.heatmap(
    order_by="age",
    order_direction="descending",
    title="Ordered by Age (Descending)",
    show_colorscale_legend=True
)
heatmap3.show()

# Example 4: Order by categorical column with custom categories
print("Example 4: Order by education level (custom category order)")
heatmap4 = missing.heatmap(
    order_by="education",
    order_categories=["Low", "Med", "High"],  # Custom order
    title="Ordered by Education (Low → Med → High)",
    show_colorscale_legend=True
)
heatmap4.show()

# Example 5: Order by two columns
print("Example 5: Order by city, then age")
heatmap5 = missing.heatmap(
    order_by=["city", "age"],
    order_direction=["ascending", "ascending"],
    title="Ordered by City, then Age",
    show_colorscale_legend=True,
    figure_size_pixels=(1000, 700)
)
heatmap5.show()

# Example 6: Mixed ordering - categorical + numeric
print("Example 6: Order by education (custom), then age (descending)")
heatmap6 = missing.heatmap(
    order_by=["education", "age"],
    order_categories=[["Low", "Med", "High"], None],  # Only first column has categories
    order_direction=["ascending", "descending"],
    title="Mixed Ordering: Education (Low→High), Age (High→Low)",
    show_colorscale_legend=True,
    figure_size_pixels=(1000, 700)
)
heatmap6.show()