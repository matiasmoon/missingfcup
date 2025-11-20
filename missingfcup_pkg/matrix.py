import pandas as pd
import plotly.graph_objects as go
from typing import Optional, List, Union

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
    """
    Interactive Missingness Matrix
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    max_cols : int, default=50
        Maximum number of columns to display.
    height, width : int
        Figure size in pixels.
    present_color, missing_color : str
        Colors for present / missing values.
    hover_template : str, optional
        Custom Plotly hover template.
    show_hover : bool
        Show hover info.
    show_scale : bool
        Show colorscale legend.
    selected_columns : list[str], optional
        Subset of columns to display.
    column_gap : float
        Space between columns.
    title : str, optional
        Custom figure title.
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive Plotly heatmap figure.
    """
    
    df = df.copy()

    if selected_columns:
        df = df[selected_columns]

    if df.shape[1] > max_cols:
        df = df.iloc[:, :max_cols]

    boolean_matrix = df.notna().astype(int)

    if hover_template is None:
        hover_template = "<b>Row</b>: %{y}<br><b>Col</b>: %{x}<br><b>Present?</b>: %{z}<extra></extra>"

    fig = go.Figure(
        go.Heatmap(
            z=boolean_matrix.values,
            x=boolean_matrix.columns,
            y=list(range(len(df))),
            colorscale=[[0, missing_color], [1, present_color]],
            showscale=show_scale,
            hoverinfo='text' if show_hover else 'skip',
            hovertemplate=hover_template,
            xgap=column_gap

        )
    )

    fig.update_layout(
        title=title if title else "Interactive Missingness Matrix",
        xaxis_title="Columns",
        yaxis_title="Row Index",
        height=height,
        width=width
    )

    return fig
