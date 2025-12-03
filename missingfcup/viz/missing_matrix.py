"""Interactive missingness matrix visualization.

This module provides the core visualization function for creating interactive
heatmaps that show which cells in a pandas DataFrame are present (1) or missing (0).
Uses Plotly for interactive, browser-based visualizations.

The `missing_matrix` function is the primary public API of the missingfcup package.
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Optional, List, Union


# Public API: missing_matrix
def missing_matrix(
    df: pd.DataFrame,
    max_cols: int = 50,
    height: int = 600,
    width: int = 900,
    present_color: str = "#2ca02c",
    missing_color: str = "#d62728",
    hover_template: Optional[str] = None,
    show_hover: bool = True,
    show_scale: bool = False,
    selected_columns: Optional[Union[List[str], None]] = None,
    column_gap: float = 0.5,
    title: Optional[str] = None,
) -> go.Figure:
    """Create an interactive missingness matrix using Plotly.

    This function converts the input DataFrame into a boolean matrix
    indicating presence (1) or missing (0) and renders it as a heatmap.

    Args:
        df: Input pandas DataFrame to visualize.
        max_cols: Maximum number of columns to display (slices wide tables).
        height: Figure height in pixels.
        width: Figure width in pixels.
        present_color: Hex color for present values (1).
        missing_color: Hex color for missing values (0).
        hover_template: Custom Plotly hover template. If None, a sensible default
            showing row index, column name and presence is used.
        show_hover: Whether to display hover information.
        show_scale: Whether to show a colorbar / scale legend.
        selected_columns: Optional list of columns to limit the display to.
        column_gap: Gap (in pixels) between columns in the heatmap.
        title: Optional title for the figure. If None, a default title is used.

    Returns:
        A `plotly.graph_objects.Figure` containing the heatmap.
    """

    # Work on a shallow copy of the dataframe so we don't mutate caller data.
    df = df.copy()

    # If the user supplied a subset of columns, select them.
    if selected_columns:
        # Filter to only columns that exist in the dataframe
        invalid_cols = [col for col in selected_columns if col not in df.columns]
        if invalid_cols:
            print(f"Warning: Columns {invalid_cols} not found in dataframe. Skipping them.")
            selected_columns = [col for col in selected_columns if col in df.columns]
        
        if not selected_columns:
            print("Warning: No valid columns provided. Using all columns.")
        else:
            df = df[selected_columns]

    # If the dataframe has more columns than `max_cols`, slice to that limit.
    if df.shape[1] > max_cols:
        df = df.iloc[:, :max_cols]

    # Create a boolean presence matrix: True where values are not NA.
    # Then convert booleans to integer 0/1 for plotting.
    boolean_matrix = df.notna().astype(int)

    # Prepare a default hover template if the caller did not provide one.
    # Plotly's hovertemplate supports %{x}, %{y}, %{z} tokens for values.
    if hover_template is None:
        hover_template = (
            "<b>Row</b>: %{y}<br>"
            "<b>Col</b>: %{x}<br>"
            "<b>Present?</b>: %{z}<extra></extra>"
        )

    # Build the heatmap trace. We pass the raw numerical matrix via `z`.
    # - `x` are the column names
    # - `y` is the row index (we use integer positions to avoid timezone/index quirks)
    # - `colorscale` maps 0->missing_color and 1->present_color
    # - `hoverinfo` and `hovertemplate` control what appears on hover
    fig = go.Figure(
        go.Heatmap(
            z=boolean_matrix.values,
            x=list(boolean_matrix.columns),
            y=list(range(len(df))),
            colorscale=[[0, missing_color], [1, present_color]],
            showscale=show_scale,
            hoverinfo="text" if show_hover else "skip",
            hovertemplate=hover_template,
            xgap=column_gap,
        )
    )

    # Set up layout with sensible default axis titles and size.
    fig.update_layout(
        title=title if title else "Interactive Missingness Matrix",
        xaxis_title="Columns",
        yaxis_title="Row Index",
        height=height,
        width=width,
    )

    # Return the assembled figure to the caller for display or further tweaks.
    return fig
