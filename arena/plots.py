"""Arena light-shift figures (moved from arena_light_shift.py)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon

from .ramsey import ramsey_no_phase


def default_styles(n):
    """``n`` style dicts from the rcParams property cycle (color, marker edge color, ...)."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()
    length = len(next(iter(cycle.values())))
    return [{key: values[i % length] for key, values in cycle.items()} for i in range(n)]


def _styles(styles, n):
    return default_styles(n) if styles is None else list(styles)


def _title(fits, i):
    name = fits.condition_names[i]
    return f"{fits.label}\n{name}" if name else fits.label


def _fit_label(fits, i, k):
    if not fits.has_fit[i, k]:
        return f"{fits.site_names[k]}: no fit"
    flag = "" if fits.reliable[i, k] else " (low contrast)"
    return f"{fits.site_names[k]}: f0 = {fits.f0[i, k]:.4f} MHz{flag}"


def _plot_data_and_fit(ax, fits, i, k, style, label=None):
    """Data points and fitted Ramsey curve of site ``k`` in condition ``i``."""
    f = fits.freqs[i]
    ax.plot(f, fits.curves[i][:, k], label=label, **{"marker": "o", "linestyle": "none", **style})
    if fits.has_fit[i, k]:
        f_fine = np.linspace(f.min(), f.max(), 1000)
        ax.plot(f_fine, ramsey_no_phase(f_fine, *fits.params[i, k]), color=style.get("color"), marker="")


def plot_site_curves(fits, *, fig=None, site_styles=None):
    """Recapture vs frequency of every site, one panel per condition."""
    styles = _styles(site_styles, len(fits.site_names))
    fig = fig or plt.figure()
    axes = fig.subplots(1, fits.n_conditions, sharex=True, sharey=True, squeeze=False)[0]
    for i, ax in enumerate(axes):
        for k, (name, style) in enumerate(zip(fits.site_names, styles)):
            ax.plot(fits.freqs[i], fits.curves[i][:, k], label=name, **{"marker": "o", **style})
        ax.set_title(fits.condition_names[i])
        ax.set_xlabel(fits.frequency_label)
    axes[0].set_ylabel("Recapture probability")
    axes[-1].legend()
    fig.suptitle(fits.label)
    return fig


def plot_arena_fits(fits, condition=0, *, fig=None, site_styles=None, ncols=4):
    """Data and fit of every site in one condition (index), one panel per site."""
    styles = _styles(site_styles, len(fits.site_names))
    fig = fig or plt.figure()
    nrows = -(-len(fits.site_names) // ncols)
    axes = fig.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= len(fits.site_names):
            ax.set_visible(False)
            continue
        _plot_data_and_fit(ax, fits, condition, k, styles[k])
        if fits.has_fit[condition, k]:
            ax.axvline(fits.f0[condition, k], color="k", linestyle="--")
        ax.set_title(_fit_label(fits, condition, k))
    for ax in axes[-1]:
        ax.set_xlabel(fits.frequency_label)
    for ax in axes[:, 0]:
        ax.set_ylabel("Recapture probability")
    fig.suptitle(_title(fits, condition))
    return fig


def plot_arena_fits_overlay(fits, condition=0, *, ax=None, site_styles=None):
    """Data and fit of every site in one condition (index), all on the same axes."""
    styles = _styles(site_styles, len(fits.site_names))
    ax = ax or plt.figure().gca()
    for k, style in enumerate(styles[:len(fits.site_names)]):
        _plot_data_and_fit(ax, fits, condition, k, style, label=_fit_label(fits, condition, k))
    ax.set_xlabel(fits.frequency_label)
    ax.set_ylabel("Recapture probability")
    ax.set_title(_title(fits, condition))
    ax.legend()
    return ax


def plot_resonance_vs_condition(fits, x=None, *, ax=None, site_styles=None):
    """Fitted f0 of every site vs a condition parameter ``x`` (reliable fits only).

    Default ``x``: the condition parameter, if there is only one; otherwise the
    conditions are placed at 0, 1, ... and labelled with their names.
    """
    styles = _styles(site_styles, len(fits.site_names))
    ax = ax or plt.figure().gca()
    keys = {key for c in fits.conditions for key in c}
    if x is None and len(keys) == 1:
        x = keys.pop()
    if x is None:
        xs = np.arange(fits.n_conditions, dtype=float)
        ax.set_xticks(xs, fits.condition_names, rotation=30, ha="right")
    else:
        xs = np.array([c[x] for c in fits.conditions])
        ax.set_xlabel(x)
    order = np.argsort(xs)
    for k, (name, style) in enumerate(zip(fits.site_names, styles)):
        m = order[fits.reliable[order, k]]
        if m.size:
            ax.errorbar(xs[m], fits.f0[m, k], yerr=fits.errors[m, k, 3], label=name, **{"marker": "o", **style})
    ax.set_ylabel("Fitted f0 (MHz)")
    ax.set_title(fits.label)
    ax.legend()
    return ax


def plot_beam_map(beam, *, fig=None, cmap=None, tile_alpha=0.6):
    """One panel per condition: arena sites as hexagons colored by their light shift.

    + arena center, ★ beam center (1σ bars), arrow from beam to arena center,
    dashed circle of radius w. Grey = no reliable shift. The site panels come
    first in ``fig.axes``, then their colorbars.
    """
    fig = fig or plt.figure()
    axes = fig.subplots(1, len(beam.conditions), squeeze=False)[0]
    xy, center, arena = beam.site_xy, beam.center, beam.arena_center
    spacing = min(np.linalg.norm(p - q) for k, p in enumerate(xy) for q in xy[k + 1:])
    tile = 0.18 * spacing  # hexagon radius
    colormap = plt.get_cmap(cmap)
    for ax, name, row in zip(axes, beam.condition_names, beam.shifts):
        finite = row[np.isfinite(row)]
        norm = plt.Normalize(finite.min(), finite.max()) if finite.size else plt.Normalize(0, 1)
        for (x, y), shift, site in zip(xy, row, beam.site_names):
            face = "lightgrey" if np.isnan(shift) else colormap(norm(shift), alpha=tile_alpha)
            ax.add_patch(RegularPolygon((x, y), 6, radius=tile, orientation=np.pi / 6,
                                        facecolor=face, edgecolor="w"))
            ax.text(x, y - 1.2 * tile, f"{site}\n{'no fit' if np.isnan(shift) else f'{shift:+.1f}'}",
                    ha="center", va="top", fontsize="small")
        ax.plot(*arena, "+", color="k", markeredgecolor="k")
        ax.errorbar(*center, xerr=beam.center_err[0], yerr=beam.center_err[1],
                    fmt="*", color="r", markeredgecolor="k")
        ax.add_patch(Circle(center, beam.waist, fill=False, edgecolor="r", linestyle="--"))
        ax.annotate("", xy=arena, xytext=center, arrowprops=dict(arrowstyle="->", color="r"))
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax,
                     label="light shift (kHz)", alpha=tile_alpha)
        ax.set_title(name)
        ax.set_xlim(arena[0] - 1.7 * spacing, arena[0] + 1.7 * spacing)
        ax.set_ylim(arena[1] - 1.7 * spacing, arena[1] + 1.7 * spacing)
        ax.set_aspect("equal")
        ax.set_xlabel(f"x ({beam.unit})")
    axes[0].set_ylabel(f"y ({beam.unit})")
    fig.suptitle(f"{beam.label}\n+ arena center, ★ beam center, "
                 "arrow = move from beam center to arena center, dashed circle = w")
    return fig


def plot_shift_vs_distance(beam, *, ax=None, condition_styles=None):
    """Measured light shift vs distance to the beam center, with the Gaussian model."""
    styles = _styles(condition_styles, len(beam.conditions))
    ax = ax or plt.figure().gca()
    r = np.linalg.norm(beam.site_xy - beam.center, axis=1)
    r_fine = np.linspace(0, 1.2 * r.max(), 200)
    for name, row, err, model, style in zip(beam.condition_names, beam.shifts, beam.shift_errors,
                                            beam.shift_model(r_fine), styles):
        ax.errorbar(r, row, yerr=err, label=name or None, **{"marker": "o", "linestyle": "none", **style})
        ax.plot(r_fine, model, color=style.get("color"), marker="")
    ax.set_xlabel(f"distance to beam center ({beam.unit})")
    ax.set_ylabel("light shift (kHz)")
    ax.set_title("Gaussian beam model vs measured shifts")
    if any(beam.condition_names):
        ax.legend()
    return ax
