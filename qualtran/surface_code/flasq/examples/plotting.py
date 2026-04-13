#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Visualization utilities for FLASQ analysis results."""
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm


def _k_formatter(x: float, pos: int) -> str:
    """Matplotlib formatter to format numbers in thousands (e.g., 10000 -> 10k)."""
    return f"{int(x/1000)}k"


def _simple_float_formatter(x: float, pos: int) -> str:
    """Matplotlib formatter to format numbers as floats with one decimal place."""
    return f"{x:.1f}"


def _scientific_formatter(x: float, pos: int) -> str:
    """Matplotlib formatter to format numbers in scientific notation."""
    return f"{x:.1e}"


def enrich_sweep_df(
    df: pd.DataFrame,
    *,
    target_std_dev: float,
) -> pd.DataFrame:
    """Adds 'Time to Solution (hr)' and 'Lambda' columns to a sweep DataFrame.

    Args:
        df: The raw DataFrame from a `run_flasq_optimization_sweep`.
        target_std_dev: The target standard deviation for the final estimate,
            used to calculate the time to solution.

    Returns:
        A new DataFrame with the added 'Time to Solution (hr)' and 'Lambda' columns.
    """
    df_copy = df.copy()
    df_copy["Time to Solution (hr)"] = (
        df_copy["Effective Time per Sample (s)"] * (target_std_dev) ** (-2) / 3600.0
    )
    df_copy["Lambda"] = 0.01 / df_copy["Physical Error Rate"]
    return df_copy


def find_optimal_heatmap_configs(
    df: pd.DataFrame,
    *,
    x_axis_col: str,
    y_axis_col: str,
    value_col_to_optimize: str,
    regularization_col: Optional[str] = None,
    regularization_strength: float = 0.01,
) -> pd.DataFrame:
    """Finds the best configuration for each point on a heatmap grid.

    From a DataFrame of sweep results, this function groups by the x and y
    axes of the heatmap and, for each grid point, finds the row that
    minimizes `value_col_to_optimize` (with an optional regularization term).
    This is primarily used to find the optimal code distance for each
    (x, y) point.

    Args:
        df: The DataFrame of sweep results. Should be filtered *before* calling.
        x_axis_col: The name of the column to use for the x-axis.
        y_axis_col: The name of the column to use for the y-axis.
        value_col_to_optimize: The column to minimize to find the best config.
        regularization_col: An optional column to add as a regularization
            term to the optimization.
        regularization_strength: The strength of the regularization.

    Returns:
        A new DataFrame containing only the optimal rows for each (x, y) point.
    """
    df_copy = df.copy()
    metric_to_min = value_col_to_optimize

    if regularization_col:
        opt_metric_name = f"__regularized_{value_col_to_optimize}"
        df_copy[opt_metric_name] = (
            df_copy[value_col_to_optimize]
            + regularization_strength * df_copy[regularization_col]
        )
        metric_to_min = opt_metric_name

    grouped = df_copy.groupby([x_axis_col, y_axis_col])
    min_indices = grouped[metric_to_min].idxmin()
    return df_copy.loc[min_indices]


def plot_flasq_heatmap(
    processed_df: pd.DataFrame,
    *,
    x_axis_col: str,
    y_axis_col: str,
    value_col_to_plot: str,
    title: str,
    log_scale: bool = True,
    tick_frequency: int = 1,
    x_formatter: Optional[Callable] = _k_formatter,
    y_formatter: Optional[Callable] = _scientific_formatter,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (7.5, 4.0),
    ax: Optional[plt.Axes] = None,
    cbar_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    skip_decimal_formatting: bool = False,
    invert_yaxis: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots a publication-quality heatmap from prepared FLASQ simulation data.

    This function encapsulates the aesthetic choices from the `ising_notebook.py`
    example to produce a standardized heatmap.

    Args:
        processed_df: The DataFrame to plot, typically the output of
            `find_optimal_heatmap_configs`.
        x_axis_col: The column for the x-axis.
        y_axis_col: The column for the y-axis.
        value_col_to_plot: The column whose values will populate the heatmap cells.
        title: The title of the plot.
        log_scale: If True, use a logarithmic color scale.
        tick_frequency: The frequency of tick labels (e.g., 1 for every tick,
            2 for every other tick).
        x_formatter: A matplotlib formatter for the x-axis tick labels.
        y_formatter: A matplotlib formatter for the y-axis tick labels.
        cmap: The colormap for the heatmap.
        figsize: The figure size.
        ax: An optional existing matplotlib Axes to plot on.
        cbar_label: An optional label for the colorbar. Defaults to `value_col_to_plot`.
        vmin: Minimum value for the color scale.
        vmax: Maximum value for the color scale.
        center: The value at which to center the colormap when plotting divergent
            data.
        skip_decimal_formatting: If True, always format annotation values
            as integers, skipping the one-decimal-place formatting for
            values below 9.95.
        invert_yaxis: If True, inverts the y-axis.

    Returns:
        A tuple of (Figure, Axes) for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    heatmap_data = processed_df.pivot_table(
        values=value_col_to_plot, index=y_axis_col, columns=x_axis_col
    )

    # Define a custom formatting function for annotations
    def format_annot(x):
        if abs(x) < 9.95 and not skip_decimal_formatting:
            return f"{x:.1f}"  # One decimal place
        else:
            return f"{x:.0f}"  # Zero decimal places (integer)

    # 1. Get axis values and format them for display.
    x_values = heatmap_data.columns.values
    y_values = heatmap_data.index.values

    formatted_x_labels = [str(x) for x in x_values]
    if x_formatter:
        formatted_x_labels = [x_formatter(x, i) for i, x in enumerate(x_values)]

    formatted_y_labels = [str(y) for y in y_values]
    if y_formatter:
        formatted_y_labels = [y_formatter(y, i) for i, y in enumerate(y_values)]

    norm = None
    if log_scale:
        norm = LogNorm(vmin=vmin, vmax=vmax)

    # Create the annotation labels by applying the custom function
    annot_labels = heatmap_data.map(format_annot)

    # 2. Draw the heatmap, passing the pre-formatted labels.
    sns.heatmap(
        heatmap_data,
        annot=annot_labels,
        fmt="",
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        center=center,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        xticklabels=formatted_x_labels,
        yticklabels=formatted_y_labels,
    )

    # 3. Customize tick labels to show only every `tick_frequency` label.
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [
            label if i % tick_frequency == 0 else ""
            for i, label in enumerate(formatted_x_labels)
        ],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [
            label if i % tick_frequency == 0 else ""
            for i, label in enumerate(formatted_y_labels)
        ],
        rotation=0,
        va="center",
    )

    # 4. Add manual grid lines and style ticks for clarity.
    ax.grid(False)
    for i, pos in enumerate(xticks):
        if i % tick_frequency == 0:
            ax.axvline(
                pos,
                color="gray",
                linestyle="--",
                linewidth=0.5,
                alpha=0.7,
                zorder=0,
            )
    for i, pos in enumerate(yticks):
        if i % tick_frequency == 0:
            ax.axhline(
                pos,
                color="gray",
                linestyle="--",
                linewidth=0.5,
                alpha=0.7,
                zorder=0,
            )

    ax.tick_params(top=False, right=False, length=5, width=1)
    if invert_yaxis:
        ax.invert_yaxis()

    ax.set_title(title)
    ax.set_xlabel(x_axis_col)
    ax.set_ylabel(y_axis_col)

    cbar = ax.collections[0].colorbar
    cbar.set_label(
        cbar_label if cbar_label is not None else value_col_to_plot,
    )

    return fig, ax
