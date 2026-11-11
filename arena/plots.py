"""Figures of the arena light-shift analysis.

Colors come from the rcParams cycle unless ``site_styles`` (a list of
``Axes.plot`` keyword dicts, reservoir first, then the arena sites) or
``style`` is given.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon

from .beam import min_site_spacing
from .ramsey import ramsey_no_phase


def default_styles(n):
    """``n`` style dicts from the rcParams property cycle (color, marker edge color, ...)."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()
    length = len(next(iter(cycle.values())))
    return [{key: values[i % length] for key, values in cycle.items()} for i in range(n)]


def _styles(styles, n):
    return default_styles(n) if styles is None else list(styles)


def _title(fits):
    return f"{fits.label}\n{fits.condition_name}" if fits.condition_name else fits.label


def _fit_label(site):
    if not site.has_fit:
        return f"{site.name}: no fit"
    flag = "" if site.reliable else " (low contrast)"
    return f"{site.name}: f0 = {site.f0_MHz:.4f} MHz{flag}"


def _plot_data_and_fit(ax, freqs_MHz, site, style, label=None):
    """Data points and fitted Ramsey curve of one site."""
    ax.plot(freqs_MHz, site.recapture, label=label, **{"marker": "o", "linestyle": "none", **style})
    if site.has_fit:
        f_fine = np.linspace(freqs_MHz.min(), freqs_MHz.max(), 1000)
        ax.plot(f_fine, ramsey_no_phase(f_fine, *site.params), color=style.get("color"), marker="")


def plot_arena_fits(fits, *, fig=None, site_styles=None, ncols=4):
    """Data and fit of every site of one ``ArenaFits``, one panel per site."""
    sites = fits.sites
    styles = _styles(site_styles, len(sites))
    fig = fig or plt.figure()
    nrows = int(np.ceil(len(sites) / ncols))
    axes = fig.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= len(sites):
            ax.set_visible(False)
            continue
        _plot_data_and_fit(ax, fits.freqs_MHz, sites[k], styles[k])
        if sites[k].has_fit:
            ax.axvline(sites[k].f0_MHz, color="k", linestyle="--")
        ax.set_title(_fit_label(sites[k]))
    for ax in axes[-1]:
        ax.set_xlabel(fits.frequency_label)
    for ax in axes[:, 0]:
        ax.set_ylabel("Recapture probability")
    fig.suptitle(_title(fits))
    return fig


def plot_arena_fits_overlay(fits, *, ax=None, site_styles=None):
    """Data and fit of every site of one ``ArenaFits``, all on the same axes."""
    sites = fits.sites
    styles = _styles(site_styles, len(sites))
    ax = ax or plt.figure().gca()
    for site, style in zip(sites, styles):
        _plot_data_and_fit(ax, fits.freqs_MHz, site, style, label=_fit_label(site))
    ax.set_xlabel(fits.frequency_label)
    ax.set_ylabel("Recapture probability")
    ax.set_title(_title(fits))
    ax.legend()
    return ax


def plot_resonance_vs_condition(fits_list, x=None, *, ax=None, site_styles=None):
    """Fitted f0 of every site vs a condition parameter ``x`` (reliable fits only).

    ``fits_list``: the list returned by ``fit_arena_scans``. Default ``x``: the
    condition parameter, if there is only one; otherwise the conditions are
    placed at 0, 1, ... and labelled with their names.
    """
    n_sites = len(fits_list[0].sites)
    styles = _styles(site_styles, n_sites)
    ax = ax or plt.figure().gca()
    keys = {key for fits in fits_list for key in fits.condition}
    if x is None and len(keys) == 1:
        x = keys.pop()
    if x is None:
        xs = np.arange(len(fits_list), dtype=float)
        ax.set_xticks(xs, [fits.condition_name for fits in fits_list], rotation=30, ha="right")
    else:
        xs = np.array([fits.condition[x] for fits in fits_list])
        ax.set_xlabel(x)
    order = np.argsort(xs)
    for k, style in enumerate(styles[:n_sites]):
        used = [i for i in order if fits_list[i].sites[k].reliable]
        if used:
            ax.errorbar(xs[used], [fits_list[i].sites[k].f0_MHz for i in used],
                        yerr=[fits_list[i].sites[k].f0_err_MHz for i in used],
                        label=fits_list[0].sites[k].name, **{"marker": "o", **style})
    ax.set_ylabel("Fitted f0 (MHz)")
    ax.set_title(fits_list[0].label)
    ax.legend()
    return ax


def plot_beam_map(beam, *, ax=None, cmap=None, tile_alpha=0.6):
    """Arena sites as hexagons colored by their light shift, with the fitted beam.

    + arena center, ★ beam center (1σ bars), arrow from beam to arena center,
    dashed circle of radius w. Grey = no reliable shift. Adds a colorbar.
    """
    ax = ax or plt.figure().gca()
    xy, center, arena = beam.site_xy_um, beam.center_um, beam.arena_center_um
    spacing = min_site_spacing(xy)
    tile = 0.18 * spacing  # hexagon radius
    colormap = plt.get_cmap(cmap)
    finite = beam.shifts_kHz[np.isfinite(beam.shifts_kHz)]
    norm = plt.Normalize(finite.min(), finite.max()) if finite.size else plt.Normalize(0, 1)
    for (x, y), shift, site in zip(xy, beam.shifts_kHz, beam.site_names):
        face = "lightgrey" if np.isnan(shift) else colormap(norm(shift), alpha=tile_alpha)
        ax.add_patch(RegularPolygon((x, y), 6, radius=tile, orientation=np.pi / 6,
                                    facecolor=face, edgecolor="w"))
        ax.text(x, y - 1.2 * tile, f"{site}\n{'no fit' if np.isnan(shift) else f'{shift:+.1f}'}",
                ha="center", va="top", fontsize="small")
    ax.plot(*arena, "+", color="k", markeredgecolor="k")
    ax.errorbar(*center, xerr=beam.center_err_um[0], yerr=beam.center_err_um[1],
                fmt="*", color="r", markeredgecolor="k")
    ax.add_patch(Circle(center, beam.waist_um, fill=False, edgecolor="r", linestyle="--"))
    ax.annotate("", xy=arena, xytext=center, arrowprops=dict(arrowstyle="->", color="r"))
    ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax,
                       label="light shift (kHz)", alpha=tile_alpha)
    ax.set_title(beam.condition_name or beam.label)
    ax.set_xlim(arena[0] - 1.7 * spacing, arena[0] + 1.7 * spacing)
    ax.set_ylim(arena[1] - 1.7 * spacing, arena[1] + 1.7 * spacing)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    return ax


def plot_shift_vs_distance(beam, *, ax=None, style=None):
    """Measured light shift vs distance to the beam center, with the Gaussian model."""
    style = default_styles(1)[0] if style is None else style
    ax = ax or plt.figure().gca()
    r = np.linalg.norm(beam.site_xy_um - beam.center_um, axis=1)
    r_fine = np.linspace(0, 1.2 * r.max(), 200)
    ax.errorbar(r, beam.shifts_kHz, yerr=beam.shift_errors_kHz, label=beam.condition_name or None,
                **{"marker": "o", "linestyle": "none", **style})
    ax.plot(r_fine, beam.shift_model_kHz(r_fine), color=style.get("color"), marker="")
    ax.set_xlabel("distance to beam center (um)")
    ax.set_ylabel("light shift (kHz)")
    ax.set_title("Gaussian beam model vs measured shifts")
    if beam.condition_name:
        ax.legend()
    return ax
