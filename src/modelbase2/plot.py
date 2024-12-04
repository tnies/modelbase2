"""Plotting Utilities Module.

This module provides functions and classes for creating various plots and visualizations
for metabolic models. It includes functionality for plotting heatmaps, time courses,
and parameter scans.

Functions:
    plot_heatmap: Plot a heatmap of the given data.
    plot_time_course: Plot a time course of the given data.
    plot_parameter_scan: Plot a parameter scan of the given data.
    plot_3d_surface: Plot a 3D surface of the given data.
    plot_3d_scatter: Plot a 3D scatter plot of the given data.
    plot_label_distribution: Plot the distribution of labels in the given data.
    plot_linear_label_distribution: Plot the distribution of linear labels in the given
        data.
    plot_label_correlation: Plot the correlation between labels in the given data.
"""

from __future__ import annotations

__all__ = [
    "FigAx",
    "FigAxs",
    "add_grid",
    "bars",
    "grid_layout",
    "heatmap",
    "heatmap_from_2d_idx",
    "heatmaps_from_2d_idx",
    "line_autogrouped",
    "line_mean_std",
    "lines",
    "lines_grouped",
    "lines_mean_std_from_2d_idx",
    "relative_label_distribution",
    "rotate_xlabels",
    "shade_protocol",
    "trajectories_2d",
    "two_axes",
    "violins",
    "violins_from_2d_idx",
]

import itertools as it
import math
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import (
    LogNorm,
    Normalize,
    SymLogNorm,
    colorConverter,  # type: ignore
)
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from modelbase2.label_map import LabelMapper

if TYPE_CHECKING:
    from matplotlib.collections import QuadMesh

    from modelbase2.linear_label_map import LinearLabelMapper
    from modelbase2.model import Model
    from modelbase2.types import Array, ArrayLike

type FigAx = tuple[Figure, Axes]
type FigAxs = tuple[Figure, list[Axes]]


##########################################################################
# Helpers
##########################################################################


def _relative_luminance(color: Array) -> float:
    """Calculate the relative luminance of a color."""
    rgb = colorConverter.to_rgba_array(color)[:, :3]

    # If RsRGB <= 0.03928 then R = RsRGB/12.92 else R = ((RsRGB+0.055)/1.055) ^ 2.4
    rsrgb = np.where(
        rgb <= 0.03928,  # noqa: PLR2004
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    # L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return np.matmul(rsrgb, [0.2126, 0.7152, 0.0722])[0]


def _get_norm(vmin: float, vmax: float) -> Normalize:
    """Get a suitable normalization object for the given data.

    Uses a logarithmic scale for values greater than 1000 or less than -1000,
    a symmetrical logarithmic scale for values less than or equal to 0,
    and a linear scale for all other values.

    Args:
        vmin: Minimum value of the data.
        vmax: Maximum value of the data.

    Returns:
        Normalize: A normalization object for the given data.

    """
    if vmax < 1000 and vmin > -1000:  # noqa: PLR2004
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif vmin <= 0:
        norm = SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax, base=10)
    else:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    return norm


def _norm_with_zero_center(df: pd.DataFrame) -> Normalize:
    """Get a normalization object with zero-centered values for the given data."""
    v = max(abs(df.min().min()), abs(df.max().max()))
    return _get_norm(vmin=-v, vmax=v)


def _partition_by_order_of_magnitude(s: pd.Series) -> list[list[str]]:
    """Partition a series into groups based on the order of magnitude of the values."""
    return [
        i.to_list()
        for i in np.floor(np.log10(s)).to_frame(name=0).groupby(0)[0].groups.values()  # type: ignore
    ]


def _split_large_groups[T](groups: list[list[T]], max_size: int) -> list[list[T]]:
    """Split groups larger than the given size into smaller groups."""
    return list(
        it.chain(
            *(
                (
                    [group]
                    if len(group) < max_size
                    else [  # type: ignore
                        list(i)
                        for i in np.array_split(group, math.ceil(len(group) / max_size))  # type: ignore
                    ]
                )
                for group in groups
            )
        )
    )  # type: ignore


def _default_color(ax: Axes, color: str | None) -> str:
    """Get a default color for the given axis."""
    return f"C{len(ax.lines)}" if color is None else color


def _default_labels(
    ax: Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
) -> None:
    """Set default labels for the given axis.

    Args:
        ax: matplotlib Axes
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        zlabel: Label for the z-axis.

    """
    ax.set_xlabel("Add a label / unit" if xlabel is None else xlabel)
    ax.set_ylabel("Add a label / unit" if ylabel is None else ylabel)
    if isinstance(ax, Axes3D):
        ax.set_zlabel("Add a label / unit" if zlabel is None else zlabel)


def _annotate_colormap(
    df: pd.DataFrame,
    ax: Axes,
    sci_annotation_bounds: tuple[float, float],
    annotation_style: str,
    hm: QuadMesh,
) -> None:
    """Annotate a heatmap with the values of the data.

    Args:
        df: Dataframe to annotate.
        ax: Axes to annotate.
        sci_annotation_bounds: Bounds for scientific notation.
        annotation_style: Style for the annotations.
        hm: QuadMesh object of the heatmap.

    """
    hm.update_scalarmappable()  # So that get_facecolor is an array
    xpos, ypos = np.meshgrid(
        np.arange(len(df.columns)),
        np.arange(len(df.index)),
    )
    for x, y, val, color in zip(
        xpos.flat,
        ypos.flat,
        hm.get_array().flat,  # type: ignore
        hm.get_facecolor(),
        strict=True,
    ):
        val_text = (
            f"{val:.{annotation_style}}"
            if sci_annotation_bounds[0] < abs(val) <= sci_annotation_bounds[1]
            else f"{val:.0e}"
        )
        ax.text(
            x + 0.5,
            y + 0.5,
            val_text,
            ha="center",
            va="center",
            color="black" if _relative_luminance(color) > 0.45 else "white",  # type: ignore  # noqa: PLR2004
        )


def add_grid(ax: Axes) -> Axes:
    """Add a grid to the given axis."""
    ax.grid(visible=True)
    ax.set_axisbelow(b=True)
    return ax


def rotate_xlabels(
    ax: Axes,
    rotation: float = 45,
    ha: Literal["left", "center", "right"] = "right",
) -> Axes:
    """Rotate the x-axis labels of the given axis.

    Args:
        ax: Axis to rotate the labels of.
        rotation: Rotation angle in degrees (default: 45).
        ha: Horizontal alignment of the labels (default

    Returns:
        Axes object for object chaining

    """
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_horizontalalignment(ha)
    return ax


##########################################################################
# General plot layout
##########################################################################


def _default_fig_ax(
    *,
    ax: Axes | None,
    grid: bool,
    figsize: tuple[float, float] | None = None,
) -> FigAx:
    """Create a figure and axes if none are provided.

    Args:
        ax: Axis to use for the plot.
        grid: Whether to add a grid to the plot.
        figsize: Size of the figure (default: None).

    Returns:
        Figure and Axes objects for the plot.

    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    else:
        fig = cast(Figure, ax.get_figure())

    if grid:
        add_grid(ax)
    return fig, ax


def _default_fig_axs(
    axs: list[Axes] | None,
    *,
    ncols: int,
    nrows: int,
    figsize: tuple[float, float] | None,
    grid: bool,
    sharex: bool,
    sharey: bool,
) -> FigAxs:
    """Create a figure and multiple axes if none are provided.

    Args:
        axs: Axes to use for the plot.
        ncols: Number of columns for the plot.
        nrows: Number of rows for the plot.
        figsize: Size of the figure (default: None).
        grid: Whether to add a grid to the plot.
        sharex: Whether to share the x-axis between the axes.
        sharey: Whether to share the y-axis between the axes.

    Returns:
        Figure and Axes objects for the plot.

    """
    if axs is None or len(axs) == 0:
        fig, axs_array = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            squeeze=False,
            layout="constrained",
        )
        axs = list(axs_array.flatten())
    else:
        fig = cast(Figure, axs[0].get_figure())

    if grid:
        for ax in axs:
            add_grid(ax)
    return fig, axs


def two_axes(
    *,
    figsize: tuple[float, float] | None = None,
    sharex: bool = True,
    sharey: bool = False,
    grid: bool = False,
) -> FigAxs:
    """Create a figure with two axes."""
    return _default_fig_axs(
        None,
        ncols=2,
        nrows=1,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        grid=grid,
    )


def grid_layout(
    n_groups: int,
    *,
    n_cols: int = 2,
    col_width: float = 3,
    row_height: float = 4,
    sharex: bool = True,
    sharey: bool = False,
    grid: bool = True,
) -> tuple[Figure, list[Axes]]:
    """Create a grid layout for the given number of groups."""
    n_cols = min(n_groups, n_cols)
    n_rows = math.ceil(n_groups / n_cols)
    figsize = (n_cols * col_width, n_rows * row_height)

    return _default_fig_axs(
        None,
        ncols=n_cols,
        nrows=n_rows,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        grid=grid,
    )


##########################################################################
# Plots
##########################################################################


def bars(
    x: pd.DataFrame,
    *,
    ax: Axes | None = None,
    grid: bool = True,
) -> FigAx:
    """Plot multiple lines on the same axis."""
    fig, ax = _default_fig_ax(ax=ax, grid=grid)
    sns.barplot(data=x, ax=ax)
    _default_labels(ax, xlabel=x.index.name, ylabel=None)
    ax.legend(x.columns)
    return fig, ax


def lines(
    x: pd.DataFrame | pd.Series,
    *,
    ax: Axes | None = None,
    grid: bool = True,
) -> FigAx:
    """Plot multiple lines on the same axis."""
    fig, ax = _default_fig_ax(ax=ax, grid=grid)
    x.plot(ax=ax)
    _default_labels(ax, xlabel=x.index.name, ylabel=None)
    ax.legend(x.columns)
    return fig, ax


def lines_grouped(
    groups: list[pd.DataFrame] | list[pd.Series],
    *,
    n_cols: int = 2,
    col_width: float = 3,
    row_height: float = 4,
    sharex: bool = True,
    sharey: bool = False,
    grid: bool = True,
) -> FigAxs:
    """Plot multiple groups of lines on separate axes."""
    fig, axs = grid_layout(
        len(groups),
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        sharex=sharex,
        sharey=sharey,
        grid=grid,
    )

    for group, ax in zip(groups, axs, strict=False):
        lines(group, ax=ax, grid=grid)

    for i in range(len(groups), len(axs)):
        axs[i].set_visible(False)

    return fig, axs


def line_autogrouped(
    s: pd.Series | pd.DataFrame,
    *,
    n_cols: int = 2,
    col_width: float = 4,
    row_height: float = 3,
    max_group_size: int = 6,
    grid: bool = True,
) -> FigAxs:
    """Plot a series or dataframe with lines grouped by order of magnitude."""
    group_names = _split_large_groups(
        _partition_by_order_of_magnitude(s)
        if isinstance(s, pd.Series)
        else _partition_by_order_of_magnitude(s.max()),
        max_size=max_group_size,
    )

    groups: list[pd.Series] | list[pd.DataFrame] = (
        [s.loc[group] for group in group_names]
        if isinstance(s, pd.Series)
        else [s.loc[:, group] for group in group_names]
    )

    return lines_grouped(
        groups,
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        grid=grid,
    )


def line_mean_std(
    df: pd.DataFrame,
    *,
    label: str | None = None,
    ax: Axes | None = None,
    color: str | None = None,
    alpha: float = 0.2,
    grid: bool = True,
) -> FigAx:
    """Plot the mean and standard deviation using a line and fill."""
    fig, ax = _default_fig_ax(ax=ax, grid=grid)
    color = _default_color(ax=ax, color=color)

    mean = df.mean(axis=1)
    std = df.std(axis=1)
    ax.plot(
        mean,
        color=color,
        label=label,
    )
    ax.fill_between(
        df.index,
        mean - std,
        mean + std,
        color=color,
        alpha=alpha,
    )
    _default_labels(ax, xlabel=df.index.name, ylabel=None)
    return fig, ax


def lines_mean_std_from_2d_idx(
    df: pd.DataFrame,
    *,
    names: list[str] | None = None,
    ax: Axes | None = None,
    alpha: float = 0.2,
    grid: bool = True,
) -> FigAx:
    """Plot the mean and standard deviation of a 2D indexed dataframe."""
    if len(cast(pd.MultiIndex, df.index).levels) != 2:  # noqa: PLR2004
        msg = "MultiIndex must have exactly two levels"
        raise ValueError(msg)

    fig, ax = _default_fig_ax(ax=ax, grid=grid)

    for name in df.columns if names is None else names:
        line_mean_std(
            df[name].unstack().T,
            label=name,
            alpha=alpha,
            ax=ax,
        )
    ax.legend()
    return fig, ax


def heatmap(
    df: pd.DataFrame,
    *,
    annotate: bool = False,
    colorbar: bool = True,
    invert_yaxis: bool = True,
    cmap: str = "RdBu_r",
    norm: Normalize | None = None,
    ax: Axes | None = None,
    cax: Axes | None = None,
    sci_annotation_bounds: tuple[float, float] = (0.01, 100),
    annotation_style: str = "2g",
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot a heatmap of the given data."""
    fig, ax = _default_fig_ax(
        ax=ax,
        figsize=(
            max(4, 0.5 * len(df.columns)),
            max(4, 0.5 * len(df.index)),
        ),
        grid=False,
    )
    if norm is None:
        norm = _norm_with_zero_center(df)

    hm = ax.pcolormesh(df, norm=norm, cmap=cmap)
    ax.set_xticks(
        np.arange(0, len(df.columns), 1) + 0.5,
        labels=df.columns,
    )
    ax.set_yticks(
        np.arange(0, len(df.index), 1) + 0.5,
        labels=df.index,
    )

    if annotate:
        _annotate_colormap(df, ax, sci_annotation_bounds, annotation_style, hm)

    if colorbar:
        # Add a colorbar
        cb = fig.colorbar(hm, cax, ax)
        cb.outline.set_linewidth(0)  # type: ignore

    if invert_yaxis:
        ax.invert_yaxis()
    rotate_xlabels(ax, rotation=45, ha="right")
    return fig, ax, hm


def heatmap_from_2d_idx(
    df: pd.DataFrame,
    variable: str,
    ax: Axes | None = None,
) -> FigAx:
    """Plot a heatmap of a 2D indexed dataframe."""
    if len(cast(pd.MultiIndex, df.index).levels) != 2:  # noqa: PLR2004
        msg = "MultiIndex must have exactly two levels"
        raise ValueError(msg)

    fig, ax = _default_fig_ax(ax=ax, grid=False)
    df2d = df[variable].unstack()

    ax.set_title(variable)
    # Note: pcolormesh swaps index/columns
    hm = ax.pcolormesh(df2d.T)
    ax.set_xlabel(df2d.index.name)
    ax.set_ylabel(df2d.columns.name)
    ax.set_xticks(
        np.arange(0, len(df2d.index), 1) + 0.5,
        labels=[f"{i:.2f}" for i in df2d.index],
    )
    ax.set_yticks(
        np.arange(0, len(df2d.columns), 1) + 0.5,
        labels=[f"{i:.2f}" for i in df2d.columns],
    )

    rotate_xlabels(ax, rotation=45, ha="right")

    # Add colorbar
    fig.colorbar(hm, ax=ax)
    return fig, ax


def heatmaps_from_2d_idx(
    df: pd.DataFrame,
    *,
    n_cols: int = 3,
    col_width_factor: float = 1,
    row_height_factor: float = 0.6,
    sharex: bool = True,
    sharey: bool = False,
) -> FigAxs:
    """Plot multiple heatmaps of a 2D indexed dataframe."""
    idx = cast(pd.MultiIndex, df.index)

    fig, axs = grid_layout(
        n_groups=len(df.columns),
        n_cols=min(n_cols, len(df)),
        col_width=len(idx.levels[0]) * col_width_factor,
        row_height=len(idx.levels[1]) * row_height_factor,
        sharex=sharex,
        sharey=sharey,
        grid=False,
    )
    for ax, var in zip(axs, df.columns, strict=False):
        heatmap_from_2d_idx(df, var, ax=ax)
    return fig, axs


def violins(
    df: pd.DataFrame,
    *,
    ax: Axes | None = None,
    grid: bool = True,
) -> FigAx:
    """Plot multiple violins on the same axis."""
    fig, ax = _default_fig_ax(ax=ax, grid=grid)
    sns.violinplot(df, ax=ax)
    _default_labels(ax=ax, xlabel="", ylabel=None)
    return fig, ax


def violins_from_2d_idx(
    df: pd.DataFrame,
    *,
    n_cols: int = 4,
    row_height: int = 2,
    sharex: bool = True,
    sharey: bool = False,
    grid: bool = True,
) -> FigAxs:
    """Plot multiple violins of a 2D indexed dataframe."""
    if len(cast(pd.MultiIndex, df.index).levels) != 2:  # noqa: PLR2004
        msg = "MultiIndex must have exactly two levels"
        raise ValueError(msg)

    fig, axs = grid_layout(
        len(df.columns),
        n_cols=n_cols,
        row_height=row_height,
        sharex=sharex,
        sharey=sharey,
        grid=grid,
    )

    for ax, col in zip(axs[: len(df.columns)], df.columns, strict=True):
        ax.set_title(col)
        violins(df[col].unstack(), ax=ax)

    for ax in axs[len(df.columns) :]:
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(0)
        ax.yaxis.set_ticks([])

    for ax in axs:
        rotate_xlabels(ax)
    return fig, axs


def shade_protocol(
    protocol: pd.Series,
    *,
    ax: Axes,
    cmap_name: str = "Greys_r",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.5,
    add_legend: bool = True,
) -> None:
    """Shade the given protocol on the given axis."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from matplotlib.legend import Legend
    from matplotlib.patches import Patch

    cmap = colormaps[cmap_name]
    norm = Normalize(
        vmin=protocol.min() if vmin is None else vmin,
        vmax=protocol.max() if vmax is None else vmax,
    )

    t0 = pd.Timedelta(seconds=0)
    for t_end, val in protocol.items():
        t_end = cast(pd.Timedelta, t_end)
        ax.axvspan(
            t0.total_seconds(),
            t_end.total_seconds(),
            facecolor=cmap(norm(val)),
            edgecolor=None,
            alpha=alpha,
        )
        t0 = t_end  # type: ignore

    if add_legend:
        ax.add_artist(
            Legend(
                ax,
                handles=[
                    Patch(
                        facecolor=cmap(norm(val)),
                        alpha=alpha,
                        label=val,
                    )  # type: ignore
                    for val in protocol
                ],
                labels=protocol,
                loc="lower right",
                bbox_to_anchor=(1.0, 0.0),
                title="protocol" if protocol.name is None else cast(str, protocol.name),
            )
        )


##########################################################################
# Plots that actually require a model :/
##########################################################################


def trajectories_2d(
    model: Model,
    x1: tuple[str, ArrayLike],
    x2: tuple[str, ArrayLike],
    y0: dict[str, float] | None = None,
    ax: Axes | None = None,
) -> FigAx:
    """Plot trajectories of two variables in a 2D phase space.

    Examples:
        >>> trajectories_2d(
        ...     model,
        ...     ("S", np.linspace(0, 1, 10)),
        ...     ("P", np.linspace(0, 1, 10)),
        ... )

    Args:
        model: Model to use for the plot.
        x1: Tuple of the first variable name and its values.
        x2: Tuple of the second variable name and its values.
        y0: Initial conditions for the model.
        ax: Axes to use for the plot.

    """
    name1, values1 = x1
    name2, values2 = x2
    n1 = len(values1)
    n2 = len(values2)
    u = np.zeros((n1, n2))
    v = np.zeros((n1, n2))
    y0 = model.get_initial_conditions() if y0 is None else y0
    for i, ii in enumerate(values1):
        for j, jj in enumerate(values2):
            rhs = model.get_right_hand_side(y0 | {name1: ii, name2: jj})
            u[i, j] = rhs[name1]
            v[i, j] = rhs[name2]

    fig, ax = _default_fig_ax(ax=ax, grid=False)
    ax.quiver(values1, values2, u.T, v.T)
    return fig, ax


##########################################################################
# Label Plots
##########################################################################


def relative_label_distribution(
    mapper: LabelMapper | LinearLabelMapper,
    concs: pd.DataFrame,
    *,
    subset: list[str] | None = None,
    n_cols: int = 2,
    col_width: float = 3,
    row_height: float = 3,
    sharey: bool = False,
    grid: bool = True,
) -> FigAxs:
    """Plot the relative distribution of labels in the given data."""
    variables = list(mapper.label_variables) if subset is None else subset
    fig, axs = grid_layout(
        n_groups=len(variables),
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        sharey=sharey,
        grid=grid,
    )
    if isinstance(mapper, LabelMapper):
        for ax, name in zip(axs, variables, strict=False):
            for i in range(mapper.label_variables[name]):
                isos = mapper.get_isotopomers_of_at_position(name, i)
                labels = cast(pd.DataFrame, concs.loc[:, isos])
                total = concs.loc[:, f"{name}__total"]
                (labels.sum(axis=1) / total).plot(ax=ax, label=f"C{i+1}")
            ax.set_title(name)
            ax.legend()
    else:
        for ax, (name, isos) in zip(
            axs, mapper.get_isotopomers(variables).items(), strict=False
        ):
            concs.loc[:, isos].plot(ax=ax)
            ax.set_title(name)
            ax.legend([f"C{i+1}" for i in range(len(isos))])

    return fig, axs
