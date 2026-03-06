import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

from missingfcup.plots.Plot import Plot
from missingfcup.core.MissingData import MissingData


class ParallelCoordinates(Plot):
    """
    Parallel coordinates plot with missing values imputed below range.

    If normalize=False, missing values are imputed to a value below the
    observed range (configurable via `impute_below_range_frac`).
    If normalize=True, missing values are placed at the bottom of the
    normalized range (0.0).

    Lines can be colored by missingness of a specific column.
    """

    def __init__(
        self,
        data: MissingData,
        *,
        selected_columns: Optional[list[str]] = None,
        missingness_color_column: Optional[str] = None,
        normalize: bool = True,
        missingness_only: bool = False,
        impute_below_range_frac: float = 0.1,
        show_colorbar: bool = False,
        line_opacity: float = 0.5,
        **kwargs,
    ):
        legend_title = kwargs.pop("legend_title", None)
        super().__init__(data=data, legend_title=legend_title, **kwargs)
        self.selected_columns = selected_columns
        self.missingness_color_column = missingness_color_column
        self.normalize = normalize
        self.missingness_only = missingness_only
        self.impute_below_range_frac = impute_below_range_frac
        self.show_colorbar = show_colorbar
        self.line_opacity = line_opacity

    def _prepare_df(self) -> pd.DataFrame:
        df = self.data.data
        if self.selected_columns is None:
            selected = df.columns.tolist()
        else:
            selected = self.selected_columns

        missing = [col for col in selected if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")

        df = df[selected]

        if not self.missingness_only:
            non_numeric = [
                col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])
            ]
            if non_numeric:
                raise TypeError(
                    "ParallelCoordinates requires numeric columns. "
                    f"Non-numeric columns: {non_numeric}"
                )

        if self.missingness_color_column is not None:
            if self.missingness_color_column not in self.data.data.columns:
                raise ValueError(
                    f"missingness_color_column '{self.missingness_color_column}' not found"
                )

        if not 0 <= self.impute_below_range_frac <= 1:
            raise ValueError("impute_below_range_frac must be between 0 and 1")

        return df

    def _normalize_series(self, series: pd.Series) -> tuple[pd.Series, float, float]:
        s = series.dropna()
        if s.empty:
            min_val, max_val = 0.0, 1.0
        else:
            min_val = float(s.min())
            max_val = float(s.max())

        span = max_val - min_val
        if span == 0:
            span = 1.0

        normalized = (series - min_val) / span
        return normalized, min_val, max_val

    def _impute_missing(self, series: pd.Series, impute_value: float) -> pd.Series:
        filled = series.copy()
        filled[self.data.missing_mask[series.name]] = impute_value
        return filled

    def _build_dimensions(self, df: pd.DataFrame) -> list[dict]:
        dimensions = []
        for col in df.columns:
            if self.missingness_only:
                values = self.data.missing_mask[col].astype(int)
                dimensions.append(
                    dict(
                        label=col,
                        values=values,
                        range=[0, 1],
                        tickvals=[0, 1],
                        ticktext=["Present", "Missing"],
                    )
                )
                continue

            if self.normalize:
                normalized, min_val, max_val = self._normalize_series(df[col])
                impute_value = 0.0
                values = self._impute_missing(normalized, impute_value)
                dim_range = [0.0, 1.0]
                tickvals = [0.0, 0.5, 1.0]
                ticktext = ["0.0", "0.5", "1.0"]
            else:
                s = df[col].dropna()
                if s.empty:
                    min_val, max_val = 0.0, 1.0
                else:
                    min_val = float(s.min())
                    max_val = float(s.max())

                span = max_val - min_val
                if span == 0:
                    span = 1.0

                impute_value = min_val - self.impute_below_range_frac * span
                values = self._impute_missing(df[col], impute_value)
                dim_range = [impute_value, max_val]
                tickvals = None
                ticktext = None

            dim = dict(
                label=col,
                values=values,
                range=dim_range,
            )
            if tickvals is not None:
                dim["tickvals"] = tickvals
                dim["ticktext"] = ticktext
            dimensions.append(dim)

        return dimensions

    def _build_figure(self) -> go.Figure:
        df = self._prepare_df()

        dimensions = self._build_dimensions(df)

        line_kwargs: dict = {}

        def apply_alpha(color: str, alpha: float) -> str:
            if color.startswith("rgba("):
                return color
            if color.startswith("rgb("):
                rgb = color[color.find("(") + 1 : color.find(")")].split(",")
                r, g, b = [int(float(x.strip())) for x in rgb[:3]]
                return f"rgba({r},{g},{b},{alpha})"
            if color.startswith("#") and len(color) in {4, 7}:
                hex_color = color.lstrip("#")
                if len(hex_color) == 3:
                    hex_color = "".join([c * 2 for c in hex_color])
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return f"rgba({r},{g},{b},{alpha})"
            return color

        if self.missingness_color_column is not None:
            missing_mask = self.data.missing_mask
            target_missing = missing_mask[self.missingness_color_column]
            line_color = target_missing.astype(int).to_numpy()
            line_kwargs.update(
                dict(
                    color=line_color,
                    colorscale=[
                        [0.0, apply_alpha(self.present_color, self.line_opacity)],
                        [1.0, apply_alpha(self.missing_color, self.line_opacity)],
                    ],
                    cmin=0,
                    cmax=1,
                    showscale=self.show_colorbar,
                    colorbar=dict(
                        title=f"{self.missingness_color_column} missingness",
                        tickmode="array",
                        tickvals=[0, 1],
                        ticktext=["Present", "Missing"],
                    ) if self.show_colorbar else None,
                )
            )
        else:
            line_kwargs.update(
                dict(color=apply_alpha(self.present_color, self.line_opacity))
            )

        fig = go.Figure(
            data=[
                go.Parcoords(
                    line=line_kwargs,
                    dimensions=dimensions,
                )
            ]
        )

        if self.missingness_color_column is not None and self.show_legend:
            present_color = apply_alpha(self.present_color, self.line_opacity)
            target_missing_color = apply_alpha(self.missing_color, self.line_opacity)

            fig.add_scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=present_color, width=6),
                name="Present",
                showlegend=True,
            )
            fig.add_scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=target_missing_color, width=6),
                name="Missing",
                showlegend=True,
            )

        self._apply_base_layout(fig)
        fig.update_layout(
            dragmode="pan",
        )

        return fig
