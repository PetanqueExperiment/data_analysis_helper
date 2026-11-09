"""Plotting functions for :class:`~data_analysis_helper.experiment.ExperimentData`.

Every function takes the experiment as its first argument and never calls
``show()``. The helpers only decide *what* is drawn (data, axis labels,
titles, legend entries). All styling (markers, line styles, colors, grid,
font sizes, figure size, legend placement, axis limits, ...) is left to the
caller, through Matplotlib rcParams / style sheets, the ``**kwargs`` forwarded
to the underlying Matplotlib call, or the returned axes.

- Single-axes plots accept ``ax`` and return it. With ``ax=None``, curves draw
  on the current axes, so successive calls overlay several experiments; maps
  of a 2-axis scan open a new figure.
- Multi-panel plots accept ``fig`` (a new figure if None) and return it.

Example::

    from data_analysis_helper.data_plotter import *

    plt.style.use("my_lab.mplstyle")       # or plt.rcParams.update({...})

    for exp in experiments:                # overlay several 1D scans
        plot_recapture(exp, marker="o")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_recapture(exp, ax=ax1)
    plot_loading(exp, ax=ax2, color="C3")

    x, y, grid = exp.compute_recapture_grid(x_axis=1)   # data behind a 2D plot
    plot_recapture_2d(exp, style="curve", x_axis=1)
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .experiment import ExperimentData

__all__ = [
    "plot_recapture",
    "plot_loading",
    "plot_fluorescence",
    "plot_fluo_recapture",
    "plot_recapture_per_atom",
    "plot_recapture_2d",
]

# Accepted spellings of the two plot kinds, mapped to their canonical name.
_STYLE_ALIASES = {
    "map": "map", "imshow": "map", "2d": "map",
    "curve": "curve", "1d": "curve", "plot": "curve",
}

# (x, y, yerr, label) for one errorbar series.
_Series = Tuple[np.ndarray, np.ndarray, np.ndarray, str]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _normalize_style(style: str) -> str:
    """Return ``"map"`` or ``"curve"`` for any accepted spelling."""
    try:
        return _STYLE_ALIASES[style.lower()]
    except KeyError:
        raise ValueError(f"Unknown style: {style!r}; use 'map' or 'curve'.") from None


def _axis_label(exp: ExperimentData, axis: int) -> str:
    """``"device parameter"`` label for scan axis ``axis``."""
    return " ".join(exp.parameters_labels[axis])


def _experiment_path(exp: ExperimentData) -> str:
    """Absolute path of the experiment folder, used in legends and titles."""
    return str(exp.experiment_folder_path)


def _finite_sorted_by_x(x: np.ndarray, y: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Drop points where ``y`` is NaN, then sort everything by ``x``.

    The sort is stable, so repeated ``x`` values keep their manifest order.
    Returns ``(x, y, *others)``.
    """
    mask = np.isfinite(y)
    order = np.argsort(x[mask], kind="stable")
    return tuple(array[mask][order] for array in (x, y, *others))


@contextmanager
def _ignore_empty_slice_warning():
    """Silence numpy's empty-slice warnings when a parameter set has no such counts."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)
        yield


def _plot_probability_curve(
    exp: ExperimentData, y: np.ndarray, ylabel: str,
    ax: Optional[Axes], label: Optional[str], plot_kwargs: dict[str, Any],
) -> Axes:
    """Shared body of the 1D recapture / loading plots."""
    if exp.scan_nb_axes < 1:
        raise ValueError("Expected at least 1 scan axis, got 0.")
    ax = plt.gca() if ax is None else ax
    x, y = _finite_sorted_by_x(exp.param_sets_array[:, 0], y)
    ax.plot(
        x, y,
        label=_experiment_path(exp) if label is None else label,
        **plot_kwargs,
    )
    ax.set_xlabel(_axis_label(exp, 0))
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def _three_panel_plot(
    fig: Optional[Figure], title: str, x_label: str,
    first: _Series, second: _Series, errorbar_kwargs: dict[str, Any],
) -> Tuple[Figure, Axes]:
    """Both series together on the left, each alone on the right (stacked).

    Each right panel reuses the color its series got on the left. Returns
    the figure and the combined (left) axes.
    """
    fig = plt.figure() if fig is None else fig
    axes = fig.subplot_mosaic([["both", "first"], ["both", "second"]])
    axes["second"].sharex(axes["first"])
    fig.suptitle(title)

    for key, (x, y, yerr, label) in (("first", first), ("second", second)):
        combined = axes["both"].errorbar(x, y, yerr=yerr, label=label, **errorbar_kwargs)
        color = combined.lines[0].get_color()
        axes[key].errorbar(x, y, yerr=yerr, **{"color": color, **errorbar_kwargs})

    axes["both"].set_ylabel("Counts")
    axes["both"].set_xlabel(x_label)
    axes["second"].set_xlabel(x_label)
    return fig, axes["both"]


# --------------------------------------------------------------------------- #
# 1D scans
# --------------------------------------------------------------------------- #

def plot_recapture(
    exp: ExperimentData, threshold: Optional[int] = None, *,
    ax: Optional[Axes] = None, label: Optional[str] = None, **plot_kwargs: Any,
) -> Axes:
    """Plot average recapture probability vs the first scan axis.

    Points with no loaded atom are skipped. ``label`` defaults to the
    absolute experiment folder path; ``threshold`` defaults to ``exp.threshold``.
    ``plot_kwargs`` go to ``Axes.plot``.
    """
    return _plot_probability_curve(
        exp, exp.compute_recapture_average(threshold), "Recapture Probability",
        ax, label, plot_kwargs,
    )


def plot_loading(
    exp: ExperimentData, threshold: Optional[int] = None, *,
    ax: Optional[Axes] = None, label: Optional[str] = None, **plot_kwargs: Any,
) -> Axes:
    """Plot loading probability (first image above threshold) vs the first scan axis.

    ``plot_kwargs`` go to ``Axes.plot``.
    """
    return _plot_probability_curve(
        exp, exp.compute_loading_array(threshold), "Loading Probability",
        ax, label, plot_kwargs,
    )


def plot_fluorescence(
    exp: ExperimentData, image_index: int, threshold: Optional[int] = None, *,
    fig: Optional[Figure] = None, **errorbar_kwargs: Any,
) -> Figure:
    """Plot mean counts above and below threshold in image ``image_index`` (0 or 1).

    Means and standard deviations are taken over all shots and atoms of each
    parameter set; the threshold is drawn as a horizontal line on the combined
    panel. ``errorbar_kwargs`` go to every ``Axes.errorbar`` call.
    """
    threshold = exp.resolve_threshold(threshold)
    data = exp.data_array[:, :, image_index, :].astype(float)
    x = exp.param_sets_array[:, 0]

    def series(selected: np.ndarray, label: str) -> _Series:
        mean = np.nanmean(selected, axis=(1, 2))
        std = np.nanstd(selected, axis=(1, 2))
        return (*_finite_sorted_by_x(x, mean, std), label)

    with _ignore_empty_slice_warning():
        above = series(np.where(data > threshold, data, np.nan), "above threshold")
        below = series(np.where(data < threshold, data, np.nan), "below threshold")

    fig, ax_both = _three_panel_plot(
        fig, f"Fluorescence Image #{image_index + 1}", _axis_label(exp, 0),
        above, below, errorbar_kwargs,
    )
    ax_both.axhline(threshold, label=f"Threshold = {threshold}")
    ax_both.legend()
    return fig


def plot_fluo_recapture(
    exp: ExperimentData, threshold: Optional[int] = None,
    threshold_loading: Optional[int] = None, threshold_recapture: Optional[int] = None,
    *, fig: Optional[Figure] = None, **errorbar_kwargs: Any,
) -> Figure:
    """Plot second-image counts, split by whether the site was loaded in the first.

    "Recaptured" is the second-image fluorescence of sites loaded in the
    first image; "Background" is that of sites that were empty. Pass either
    ``threshold`` (or nothing, for ``exp.threshold``), or both
    ``threshold_loading`` and ``threshold_recapture``. ``errorbar_kwargs``
    go to every ``Axes.errorbar`` call.
    """
    if threshold_loading is None and threshold_recapture is None:
        threshold_loading = exp.resolve_threshold(threshold)
        threshold_recapture = threshold_loading
    elif threshold_loading is None or threshold_recapture is None:
        raise TypeError("Both or no recapture and loading threshold need to be provided")

    loaded = exp.data_array[:, :, 0, :] > threshold_loading
    second_image = exp.data_array[:, :, 1, :]
    x = exp.param_sets_array[:, 0]

    def series(selected: np.ndarray, label: str) -> _Series:
        mean = np.nanmean(selected, axis=(1, 2))
        std = np.nanstd(selected, axis=(1, 2))
        return (*_finite_sorted_by_x(x, mean, std), label)

    with _ignore_empty_slice_warning():
        recaptured = series(np.where(loaded, second_image, np.nan), "Recaptured")
        background = series(np.where(~loaded, second_image, np.nan), "Background")

    fig, ax_both = _three_panel_plot(
        fig, f"Recapture fluo, {_experiment_path(exp)}", _axis_label(exp, 0),
        recaptured, background, errorbar_kwargs,
    )
    ax_both.legend()
    return fig


def plot_recapture_per_atom(
    exp: ExperimentData, threshold: Optional[int] = None, *,
    style: str = "map", ax: Optional[Axes] = None, **plot_kwargs: Any,
) -> Axes:
    """Plot recapture probability of each atom (ROI) vs the single scan axis.

    ``style="map"`` (or ``"imshow"``, ``"2d"``) draws an atom-index × scan
    heatmap with a colorbar (``plot_kwargs`` go to ``Axes.pcolormesh``);
    ``style="curve"`` (or ``"1d"``, ``"plot"``) draws one curve per atom
    (``plot_kwargs`` go to ``Axes.plot``). Draws on the current axes when
    ``ax`` is None.
    """
    if exp.scan_nb_axes != 1:
        raise ValueError(f"Expected 1 scan axis, got {exp.scan_nb_axes}")
    style = _normalize_style(style)
    recapture_per_atom = exp.compute_recapture_per_atom(threshold)
    ax = plt.gca() if ax is None else ax
    order = np.argsort(exp.param_sets_array[:, 0])
    x = exp.param_sets_array[order, 0]
    recapture_per_atom = recapture_per_atom[order]

    if style == "map":
        mesh = ax.pcolormesh(x, np.arange(exp.atom_number), recapture_per_atom.T, **plot_kwargs)
        ax.figure.colorbar(mesh, ax=ax, label="Recapture Probability")
        ax.set_ylabel("Atom index")
    else:
        ax.plot(x, recapture_per_atom, **plot_kwargs)  # one line per column/atom
        ax.set_ylabel("Recapture Probability")

    ax.set_xlabel(_axis_label(exp, 0))
    ax.set_title(_experiment_path(exp))
    return ax


# --------------------------------------------------------------------------- #
# 2D scans
# --------------------------------------------------------------------------- #

def plot_recapture_2d(
    exp: ExperimentData, threshold: Optional[int] = None, *,
    style: str = "map", x_axis: int = 0, ax: Optional[Axes] = None, **plot_kwargs: Any,
) -> Axes:
    """Plot the average recapture of a 2-axis scan.

    ``style="map"`` (or ``"imshow"``, ``"2d"``) draws a heatmap with a colorbar
    (``plot_kwargs`` go to ``Axes.pcolormesh``); ``style="curve"`` (or ``"1d"``,
    ``"plot"``) draws one recapture curve per value of the other axis, with
    gaps at missing points (``plot_kwargs`` go to ``Axes.plot``). ``x_axis``
    (0 or 1) is the scan axis drawn horizontally. Opens a new figure when
    ``ax`` is None.
    """
    style = _normalize_style(style)
    x_vals, y_vals, grid = exp.compute_recapture_grid(threshold, x_axis=x_axis)
    if ax is None:
        _, ax = plt.subplots()
    x_label = _axis_label(exp, x_axis)
    y_label = _axis_label(exp, 1 - x_axis)

    if style == "curve":
        for y, row in zip(y_vals, grid):
            ax.plot(x_vals, row, label=f"{y:g}", **plot_kwargs)
        ax.set_ylabel("Recapture Probability")
        ax.legend(title=y_label)
    else:
        mesh = ax.pcolormesh(x_vals, y_vals, grid, **plot_kwargs)
        ax.figure.colorbar(mesh, ax=ax, label="Recapture Probability")
        ax.set_ylabel(y_label)

    ax.set_xlabel(x_label)
    ax.set_title(_experiment_path(exp))
    return ax
