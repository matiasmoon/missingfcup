import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Literal

from .plot import Plot
from ..core.matrix import Matrix

class Heatmap(Plot):
    """
    Interactive missingness heatmap visualization.

    Shows a grid where each cell represents whether data is present (green) or missing (red).
    Rows are data records, columns are variables from your dataset.

    Examples
    --------
    Basic usage with default settings:

    >>> heatmap = Heatmap(matrix)
    >>> heatmap.show()

    Customized visualization:

    >>> heatmap = Heatmap(
    ...     matrix,
    ...     figure_size_pixels=(1200, 800),      # larger figure
    ...     present_color="#4CAF50",              # custom green
    ...     missing_color="#FF5722",              # custom red
    ...     show_colorscale_legend=True,          # show the legend
    ...     column_spacing=1.0,                   # more space between columns
    ...     title="My Data Completeness"         # custom title
    ... )
    >>> heatmap.show()

    Order rows by a column to reveal patterns:

    >>> heatmap = Heatmap(
    ...     matrix,
    ...     order_by="age",                       # order rows by age column
    ...     order_direction="ascending"           # lowest to highest
    ... )
    >>> heatmap.show()
    """

    def __init__(
        self,
        matrix: Matrix,
        figure_size_pixels: tuple[int, int] = (900, 600),
        present_color: str = "#2ca02c",
        missing_color: str = "#d62728",
        show_hover_info: bool = True,
        custom_hover_template: Optional[str] = None,
        show_colorscale_legend: bool = False,
        column_spacing: float = 0.5,
        title: Optional[str] = None,
        order_by: Optional[Union[str, List[str]]] = None,
        order_direction: Union[Literal["ascending", "descending"], List[Literal["ascending", "descending"]]] = "ascending",
        order_categories: Optional[Union[List, List[List]]] = None,
    ):
        """
        Create an interactive missingness heatmap.

        Parameters
        ----------
        matrix : Matrix
            The missingness matrix to visualize. Create this using Matrix.from_dataframe().

        figure_size_pixels : tuple[int, int], default=(900, 600)
            Figure size as (width, height) in pixels.
            Example: (1200, 800) for a larger plot.

        present_color : str, default="#2ca02c"
            Color for cells with data present (hex color code).
            Default is green. Try "#4CAF50" for a brighter green.

        missing_color : str, default="#d62728"
            Color for cells with missing data (hex color code).
            Default is red. Try "#FF5722" for a brighter red.

        show_hover_info : bool, default=True
            Whether to show information when hovering over cells.
            When True, displays row index, column name, and presence status.
            Hover interactions include zoom and pan capabilities.

        custom_hover_template : str, optional
            Custom HTML template for hover tooltips.
            If None, uses default template showing row, column, and presence.
            Advanced users can customize this with Plotly template syntax.

        show_colorscale_legend : bool, default=False
            Whether to show the colorscale legend on the right side.
            Legend shows the mapping: 0 = missing (red), 1 = present (green).

        column_spacing : float, default=0.5
            Space between columns in pixels.
            Increase for more visual separation (e.g., 1.0 or 2.0).
            Set to 0 for no gaps.

        title : str, optional
            Custom title for the plot.
            If None, uses "Interactive Missingness Matrix" as default.

        order_by : str or list of str, optional
            Column name(s) to sort rows by. Maximum of 2 columns allowed.
            When specified, rows will be reordered to reveal patterns in missing data.
            The sorted column(s) will be visually highlighted in the plot.

            Examples:
            - "age" - sort by single column
            - ["city", "age"] - sort by city first, then age

        order_direction : {"ascending", "descending"} or list of them, default="ascending"
            Direction to sort numeric columns.

            For a single column:
            - "ascending": lowest to highest (default)
            - "descending": highest to lowest

            For multiple columns, provide a list:
            - ["ascending", "descending"] - first column ascending, second descending

            For categorical columns, use `order_categories` instead.

        order_categories : list or list of lists, optional
            Custom ordering for categorical (non-numeric) columns.
            Only used when sorting by categorical columns.

            For a single categorical column:
            - ["Low", "Medium", "High"] - order categories in this sequence

            For multiple columns where at least one is categorical:
            - [["Small", "Large"], None] - first column uses custom order, second uses default

            If not provided for a categorical column, pandas default ordering is used.

        Notes
        -----
        Column Selection:
        To control which columns appear in the heatmap, configure the Matrix
        when creating it using Matrix.from_dataframe():

        - max_cols: Maximum number of columns to display (default=50)
        - selected_columns: Manually choose specific columns by name
        - completeness_mode: Select "top" (most complete) or "bottom" (least complete)
        - completeness_threshold: Filter columns by completeness percentage
        - max_columns_by_completeness: Limit to N most/least complete columns

        Example:
        >>> matrix = Matrix.from_dataframe(
        ...     df,
        ...     max_cols=30,                      # show at most 30 columns
        ...     completeness_mode="top",          # prioritize complete columns
        ...     completeness_threshold=0.8        # only columns ≥80% complete
        ... )

        Nullity Behavior:
        The matrix uses .notna() to detect presence: True (1) means data exists,
        False (0) means NaN/None/missing. All pandas null types are treated as missing.
        """
        self.matrix = matrix
        self.width, self.height = figure_size_pixels
        self.present_color = present_color
        self.missing_color = missing_color
        self.show_hover = show_hover_info
        self.hover_template = custom_hover_template
        self.show_scale = show_colorscale_legend
        self.column_gap = column_spacing
        self.title = title
        self.order_by = order_by
        self.order_direction = order_direction
        self.order_categories = order_categories

        # Validate ordering parameters
        if self.order_by is not None:
            self._validate_ordering_params()

        self._figure: Optional[go.Figure] = None

    def _validate_ordering_params(self):
        """Validate ordering parameters."""
        if self.matrix.dataframe is None:
            raise ValueError(
                "Cannot use order_by: Matrix does not contain original DataFrame. "
                "Make sure the Matrix was created with Matrix.from_dataframe()."
            )

        # Normalize order_by to list
        order_cols = [self.order_by] if isinstance(self.order_by, str) else self.order_by

        # Check maximum 2 columns
        if len(order_cols) > 2:
            raise ValueError(
                f"Maximum 2 columns allowed for ordering. Got {len(order_cols)}: {order_cols}"
            )

        # Check columns exist in dataframe
        df = self.matrix.dataframe
        missing_cols = [col for col in order_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Ordering columns not found in DataFrame: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

    def _get_sorted_data(self):
        """Sort matrix data according to order_by parameters and return sorted values, y-labels, and order columns."""
        if self.order_by is None:
            # No ordering - return original data
            return self.matrix.values, list(range(len(self.matrix.values))), None

        df = self.matrix.dataframe.copy()
        order_cols = [self.order_by] if isinstance(self.order_by, str) else self.order_by

        # Normalize direction to list
        if isinstance(self.order_direction, str):
            directions = [self.order_direction] * len(order_cols)
        else:
            directions = self.order_direction
            if len(directions) != len(order_cols):
                raise ValueError(
                    f"Number of order_direction ({len(directions)}) must match "
                    f"number of order_by columns ({len(order_cols)})"
                )

        # Normalize categories to list
        if self.order_categories is None:
            categories_list = [None] * len(order_cols)
        elif isinstance(self.order_categories[0], list) if self.order_categories else False:
            categories_list = self.order_categories
        else:
            categories_list = [self.order_categories]

        if len(categories_list) != len(order_cols):
            raise ValueError(
                f"Number of order_categories ({len(categories_list)}) must match "
                f"number of order_by columns ({len(order_cols)})"
            )

        # Build sort keys for each column
        sort_by = []
        sort_ascending = []

        for col, direction, categories in zip(order_cols, directions, categories_list):
            if categories is not None:
                # Categorical ordering with custom categories
                cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
                df[f"{col}_sort"] = df[col].astype(cat_type)
                sort_by.append(f"{col}_sort")
                sort_ascending.append(True)  # Categories already define order
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Numeric ordering
                sort_by.append(col)
                sort_ascending.append(direction == "ascending")
            else:
                # Default categorical ordering (pandas default)
                sort_by.append(col)
                sort_ascending.append(direction == "ascending")

        # Sort the dataframe
        df_sorted = df.sort_values(by=sort_by, ascending=sort_ascending)

        # Get the sorted indices
        sorted_indices = df_sorted.index.tolist()

        # Sort the matrix values accordingly
        sorted_matrix = self.matrix.values[sorted_indices, :]

        # Create y-axis labels from the ordering column(s)
        y_labels = []
        for idx in sorted_indices:
            label_parts = [str(df.loc[idx, col]) for col in order_cols]
            y_labels.append(" | ".join(label_parts))

        return sorted_matrix, y_labels, order_cols

    def _build_figure(self) -> go.Figure:
        # Get sorted data if ordering is specified
        sorted_matrix, y_labels, order_cols = self._get_sorted_data()

        hover_template = self.hover_template or (
            "<b>Row</b>: %{y}<br>"
            "<b>Col</b>: %{x}<br>"
            "<b>Present?</b>: %{z}<extra></extra>"
        )

        fig = go.Figure(
            go.Heatmap(
                z=sorted_matrix,
                x=self.matrix.columns,
                y=y_labels,
                colorscale=[[0, self.missing_color], [1, self.present_color]],
                showscale=self.show_scale,
                hoverinfo="text" if self.show_hover else "skip",
                hovertemplate=hover_template,
                xgap=self.column_gap,
            )
        )

        # Determine y-axis title
        if order_cols:
            yaxis_title = " | ".join(order_cols)
        else:
            yaxis_title = "Row Index"

        fig.update_layout(
            title=self.title or "Interactive Missingness Matrix",
            xaxis_title="Columns",
            yaxis_title=yaxis_title,
            height=self.height,
            width=self.width,
        )

        # Highlight ordered columns with visual indication
        if order_cols:
            # Find indices of ordered columns in the matrix
            ordered_indices = [i for i, col in enumerate(self.matrix.columns) if col in order_cols]

            # Add visual highlighting: add a subtle border/emphasis to ordered columns
            for idx in ordered_indices:
                # Add a shape (rectangle) to highlight the column
                fig.add_shape(
                    type="rect",
                    x0=idx - 0.5,
                    x1=idx + 0.5,
                    y0=-0.5,
                    y1=len(y_labels) - 0.5,
                    line=dict(color="#FFA500", width=3),  # Orange border
                    fillcolor="rgba(0,0,0,0)",  # Transparent fill
                    layer="above"
                )

                # Add annotation at the top to mark the ordered column
                fig.add_annotation(
                    x=idx,
                    y=len(y_labels),
                    text=f"📊",  # Chart emoji to indicate ordering
                    showarrow=False,
                    font=dict(size=16),
                    yshift=10
                )

        return fig

    @property
    def fig(self) -> go.Figure:
        """Get or build the figure."""
        if self._figure is None:
            self._figure = self._build_figure()
        return self._figure

    def show(self):
        if self._figure is None:
            self._figure = self._build_figure()
        self._figure.show()

    def save(self, path: str):
        if self._figure is None:
            self._figure = self._build_figure()
        self._figure.write_html(path)
