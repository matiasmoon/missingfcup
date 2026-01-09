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

    Group rows by a column to reveal patterns:

    >>> heatmap = Heatmap(
    ...     matrix,
    ...     group_by="age",                       # group rows by age column
    ...     group_direction="ascending"           # lowest to highest
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
        sort_by_columns: Optional[Union[str, List[str]]] = None,
        sort_direction: Union[Literal["ascending", "descending"], List[Literal["ascending", "descending"]]] = "ascending",
        sort_categorical: Optional[Union[List, List[List]]] = None,
        group_by_mode: Literal["binary", "completeness"] = "binary",
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
            Legend shows discrete values: "Missing" (red) and "Present" (green).
            The legend displays only these two distinct colors, not a gradient.

        column_spacing : float, default=0.5
            Space between columns in pixels.
            Increase for more visual separation (e.g., 1.0 or 2.0).
            Set to 0 for no gaps.

        title : str, optional
            Custom title for the plot.
            If None, uses "Interactive Missingness Matrix" as default.

        sort_by_columns : str or list of str, optional
            Column name(s) to sort rows by. Maximum of 2 columns allowed.
            When specified, rows will be grouped and reordered to reveal patterns.
            The grouping column(s) will be moved to the left and visually highlighted.

            If not specified (default), the Y-axis will show values from the first
            (leftmost) column in the matrix, in their original order.

            Examples:
            - "age" - sort by single column
            - ["city", "age"] - sort by city first, then age

        sort_direction : {"ascending", "descending"} or list of them, default="ascending"
            Direction to sort numeric sorting columns.

            For a single column:
            - "ascending": lowest to highest (default)
            - "descending": highest to lowest

            For multiple columns, provide a list:
            - ["ascending", "descending"] - first column ascending, second descending

            For categorical columns, use `sort_categorical` instead.

        sort_categorical : list or list of lists, optional
            Custom ordering for categorical (non-numeric) sorting columns.
            Only used when sorting by categorical columns.

            For a single categorical column:
            - ["Low", "Medium", "High"] - order categories in this sequence

            For multiple columns where at least one is categorical:
            - [["Small", "Large"], None] - first column uses custom order, second uses default

            If not provided for a categorical column, pandas default ordering is used.

            Note: Missing values (NaN/None) in the sorting column will appear as "nan".

        group_by_mode : {"binary", "completeness"}, default="binary"
            How to visualize missingness when using sort_by_columns.

            - "binary": Show presence (green) or absence (red) for each cell (default).
              Uses the two discrete colors specified by present_color and missing_color.
              Hover shows "Present?: 0 or 1".

            - "completeness": Show cell-level missingness as a percentage (0% or 100%).
              Each cell is colored based on whether the value is present (0% missing, green)
              or missing (100% missing, red).
              This mode uses the same color scale as binary but displays values as percentages.
              Hover shows "Missing: 0% or 100%".

            Note: This parameter only affects visualization when sort_by_columns is specified.
            Without sort_by_columns, the heatmap always uses binary mode.

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

        Y-Axis Behavior:
        - Without group_by: Shows values from the first (leftmost) column in original order
        - With group_by: Shows values from the specified grouping column(s), sorted
        - Grouping columns are always displayed on the left side of the heatmap
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
        self.sort_by_columns = sort_by_columns
        self.sort_direction = sort_direction
        self.sort_categorical = sort_categorical
        self.group_by_mode = group_by_mode

        # Validate grouping parameters
        if self.sort_by_columns is not None:
            self._validate_sorting_params()

        self._figure: Optional[go.Figure] = None

    def _validate_sorting_params(self):
        """Validate sorting parameters."""
        if self.matrix.dataframe is None:
            raise ValueError(
                "Cannot use sort_by_columns: Matrix does not contain original DataFrame. "
                "Make sure the Matrix was created with Matrix.from_dataframe()."
            )
    
        # Normalize sort_by_columns to list
        sort_cols = [self.sort_by_columns] if isinstance(self.sort_by_columns, str) else self.sort_by_columns

        # Check maximum 2 columns
        if len(sort_cols) > 2:
            raise ValueError(
                f"Maximum 2 columns allowed for sorting. Got {len(sort_cols)}: {sort_cols}"
            )

        # Check columns exist in dataframe
        df = self.matrix.dataframe
        missing_cols = [col for col in sort_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Sorting columns not found in DataFrame: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

    def _get_sorted_data(self):
        """Sort matrix data according to sort_by_columns parameters and return sorted values, y-labels, and group columns."""
        if self.sort_by_columns is None:
            # No grouping - use first column values for Y-axis labels
            if self.matrix.dataframe is not None and len(self.matrix.dataframe.columns) > 0:
                first_col = self.matrix.dataframe.columns[0]
                y_labels = [str(val) for val in self.matrix.dataframe[first_col].values]
                return self.matrix.values, y_labels, None
            else:
                # Fallback to row indices if no dataframe available
                return self.matrix.values, list(range(len(self.matrix.values))), None

        df = self.matrix.dataframe.copy()
        sort_cols = [self.sort_by_columns] if isinstance(self.sort_by_columns, str) else self.sort_by_columns

        # Normalize direction to list
        if isinstance(self.sort_direction, str):
            directions = [self.sort_direction] * len(sort_cols)
        else:
            directions = self.sort_direction
            if len(directions) != len(sort_cols):
                raise ValueError(
                    f"Number of sort_direction ({len(directions)}) must match "
                    f"number of sort_by_columns ({len(sort_cols)})"
                )

        # Normalize categories to list
        if self.sort_categorical is None:
            categories_list = [None] * len(sort_cols)
        elif isinstance(self.sort_categorical[0], list) if self.sort_categorical else False:
            categories_list = self.sort_categorical
        else:
            categories_list = [self.sort_categorical]

        if len(categories_list) != len(sort_cols):
            raise ValueError(
                f"Number of sort_categorical ({len(categories_list)}) must match "
                f"number of sort_by_columns ({len(sort_cols)})"
            )

        # Build sort keys for each column
        sort_by = []
        sort_ascending = []

        for col, direction, categories in zip(sort_cols, directions, categories_list):
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

        # Create y-axis labels from the grouping column(s)
        y_labels = []
        for idx in sorted_indices:
            label_parts = [str(df.loc[idx, col]) for col in sort_cols]
            y_labels.append(" | ".join(label_parts))

        return sorted_matrix, y_labels, sort_cols

    def _build_figure(self) -> go.Figure:
        # Get sorted data if grouping is specified
        sorted_matrix, y_labels, sort_cols = self._get_sorted_data()

        # Reorder columns so that sort_by columns are on the left
        if sort_cols:
            # Get indices of sorting columns IN THE ORDER SPECIFIED
            sort_indices = []
            for sort_col in sort_cols:
                for i, col in enumerate(self.matrix.columns):
                    if col == sort_col:
                        sort_indices.append(i)
                        break

            # Get indices of other columns
            other_indices = [i for i, col in enumerate(self.matrix.columns) if col not in sort_cols]
            # Combine: sorting columns first (in specified order), then others
            column_order = sort_indices + other_indices

            # Reorder matrix columns and column names
            sorted_matrix_cols = sorted_matrix[:, column_order]
            reordered_columns = [self.matrix.columns[i] for i in column_order]
        else:
            sorted_matrix_cols = sorted_matrix
            reordered_columns = self.matrix.columns

        # Determine if we're using completeness mode
        use_completeness = self.group_by_mode == "completeness" and self.sort_by_columns is not None

        if use_completeness:
            # Calculate cell-level missingness (0 = present, 1 = missing)
            # Invert the binary matrix: 1 becomes 0 (present), 0 becomes 1 (missing)
            z_data = 1 - sorted_matrix_cols

            # Use a continuous colorscale from present_color (0% missing) to missing_color (100% missing)
            colorscale = [[0, self.present_color], [1, self.missing_color]]

            hover_template = self.hover_template or (
                "<b>Row</b>: %{y}<br>"
                "<b>Col</b>: %{x}<br>"
                "<b>Missing</b>: %{z:.0%}<extra></extra>"
            )

            colorbar_config = dict(
                title="% Missing",
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['0%', '100%'],
                len=0.3,
            ) if self.show_scale else None
        else:
            # Binary mode (default)
            z_data = sorted_matrix_cols
            colorscale = [[0, self.missing_color], [1, self.present_color]]

            hover_template = self.hover_template or (
                "<b>Row</b>: %{y}<br>"
                "<b>Col</b>: %{x}<br>"
                "<b>Present?</b>: %{z}<extra></extra>"
            )

            colorbar_config = dict(
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['Missing', 'Present'],
                len=0.3,
            ) if self.show_scale else None

        fig = go.Figure(
            go.Heatmap(
                z=z_data,
                x=reordered_columns,
                y=y_labels,
                colorscale=colorscale,
                showscale=self.show_scale,
                hoverinfo="text" if self.show_hover else "skip",
                hovertemplate=hover_template,
                xgap=self.column_gap,
                colorbar=colorbar_config,
                zmin=0,
                zmax=1,
            )
        )

        fig.update_layout(
            title=self.title or "Interactive Missingness Matrix",
            xaxis_title="",  # No default x-axis title
            yaxis_title="",  # No default y-axis title
            height=self.height,
            width=self.width,
        )

        # Highlight sorting columns with visual indication
        if sort_cols:
            # Sort_by columns are now at the left (indices 0, 1)
            # Add visual highlighting for each sort_by column
            for idx in range(len(sort_cols)):
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
