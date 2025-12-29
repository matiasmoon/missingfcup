import plotly.graph_objects as go
from typing import Optional

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

        self._figure: Optional[go.Figure] = None

    def _build_figure(self) -> go.Figure:
        hover_template = self.hover_template or (
            "<b>Row</b>: %{y}<br>"
            "<b>Col</b>: %{x}<br>"
            "<b>Present?</b>: %{z}<extra></extra>"
        )

        fig = go.Figure(
            go.Heatmap(
                z=self.matrix.values,
                x=self.matrix.columns,
                y=list(range(len(self.matrix.values))),
                colorscale=[[0, self.missing_color], [1, self.present_color]],
                showscale=self.show_scale,
                hoverinfo="text" if self.show_hover else "skip",
                hovertemplate=hover_template,
                xgap=self.column_gap,
            )
        )

        fig.update_layout(
            title=self.title or "Interactive Missingness Matrix",
            xaxis_title="Columns",
            yaxis_title="Row Index",
            height=self.height,
            width=self.width,
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
