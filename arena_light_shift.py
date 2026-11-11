"""Arena light-shift analysis from a Ramsey scan of arena vs reservoir sites.

The last ``n_arena`` ROIs are the arena sites (a ring of 6 plus its center);
all other ROIs form the reservoir, which sees no arena light and gives the
reference resonance.

The scan has one microwave frequency axis and any number of other axes
(arena amplitude, grating, repetition, ...). Each combination of the other
axes is a *condition*, stored as a dict ``{"device parameter": value}``.

Typical use::

    from data_analysis_helper import arena_light_shift as als

    fits = als.fit_arena_scans(exp, n_arena=7)          # Ramsey fit of every site
    als.plot_arena_fits_overlay(fits, 0)                # data + fits of condition 0
    beam = als.fit_beam_center(fits, als.arena_positions(exp))
    print(beam.summary())
    als.plot_beam_map(beam)

Or everything at once (fits, inhomogeneity, beam, next grating, one figure)::

    result = als.analyze_arena_ramsey(exp, grating=(41.38, 51.18))  # positions from TRAP_FILE

Units: frequency MHz, Ramsey time T μs, light shifts kHz. Positions are
those given to ``fit_beam_center``: SLM trap-file positions in um
(``trap_file_positions``, the default of ``analyze_arena_ramsey``) or camera
ROI pixels (``arena_positions``). Colors come from the rcParams cycle unless ``site_styles`` /
``condition_styles`` (lists of ``Axes.plot`` keyword dicts) are given.
"""

import json
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from scipy.optimize import curve_fit

from .models import ramsey_spectro

RESERVOIR = "Reservoir"


# --------------------------------------------------------------------------- #
# Sites
# --------------------------------------------------------------------------- #

def site_names(n_arena):
    """``["Reservoir", "Arena 0", ..., "Arena n-1"]``."""
    return [RESERVOIR] + [f"Arena {i}" for i in range(n_arena)]


def arena_positions(exp, n_arena=7):
    """ROI centers (px) of the arena sites, shape ``(n_arena, 2)``, from ``rois.json``."""
    with open(exp.experiment_folder_path / "rois.json", encoding="utf-8") as f:
        return np.array(json.load(f)["centers"], dtype=float)[-n_arena:]


def trap_file_positions(path, n_arena=7):
    """Design positions (um) of the arena sites, shape ``(n_arena, 2)``, from an SLM trap file.

    The last ``n_arena`` traps of the file, in the same order as the arena ROIs.
    """
    with open(path, encoding="utf-8") as f:
        return np.array(json.load(f)["centers"], dtype=float)[-n_arena:]


def default_styles(n):
    """``n`` style dicts from the rcParams property cycle (color, marker edge color, ...)."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()
    length = len(next(iter(cycle.values())))
    return [{key: values[i % length] for key, values in cycle.items()} for i in range(n)]


def site_curves(sub_exp, n_arena):
    """``(freqs, curves)`` of a 1-axis frequency scan, sorted by frequency.

    ``curves[:, 0]`` is the mean recapture of the reservoir sites,
    ``curves[:, 1:]`` the recapture of each arena site.
    """
    recapture = sub_exp.compute_recapture_per_atom()
    with warnings.catch_warnings():  # reservoir sites never loaded are NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        reservoir = np.nanmean(recapture[:, :-n_arena], axis=1)
    curves = np.column_stack([reservoir, recapture[:, -n_arena:]])
    freqs = sub_exp.param_sets_array[:, 0]
    order = np.argsort(freqs)
    return freqs[order], curves[order]


# --------------------------------------------------------------------------- #
# Ramsey fits
# --------------------------------------------------------------------------- #

def ramsey_no_phase(f, offset, amplitude, f_rabi, f_resonance, T):
    """Ramsey lineshape (``models.ramsey_spectro``) with the phase fixed to 0."""
    return ramsey_spectro(f, offset, amplitude, f_rabi, f_resonance, T, phi=0)


def fit_ramsey(f, y, T_max=25, f_rabi_starts=(0.01, 0.02), n_f0_starts=10, min_contrast=0.05):
    """Fit ``ramsey_no_phase`` from a grid of starting points; keep the best fit.

    Returns ``(params, errors)`` (offset, amplitude, f_rabi, f_resonance, T),
    or ``(None, None)`` for a flat curve, too few points, or no convergence.
    ``T`` is bounded by ``T_max`` so the fit cannot jump to an aliased fringe.
    """
    ok = np.isfinite(y)
    f, y = f[ok], y[ok]
    if len(f) < 6 or np.ptp(y) < min_contrast:
        return None, None
    bounds = ([-0.1, 0, 1e-3, f.min(), 0], [1, 2, 0.5, f.max(), T_max])
    margin = 0.2 * np.ptp(f)
    best_residual, best = np.inf, (None, None)
    for f_rabi in f_rabi_starts:
        for f0 in np.linspace(f.min() + margin, f.max() - margin, n_f0_starts):
            try:
                p, cov = curve_fit(ramsey_no_phase, f, y, p0=[y.min(), np.ptp(y), f_rabi, f0, T_max / 2],
                                   bounds=bounds, maxfev=5000)
            except (RuntimeError, ValueError):
                continue
            residual = np.sum((ramsey_no_phase(f, *p) - y) ** 2)
            if residual < best_residual:
                best_residual, best = residual, (p, np.sqrt(np.diag(cov)))
    return best


@dataclass
class ArenaFits:
    """Ramsey fits of every site, for every condition.

    ``params`` / ``errors`` have shape ``(n_conditions, n_sites, 5)``, sites in
    ``site_names`` order (reservoir first), NaN where there is no fit.
    """

    label: str               # experiment folder
    frequency_label: str
    conditions: list         # one {"device parameter": value} dict per condition
    site_names: list
    freqs: list              # per condition, sorted frequencies
    curves: list             # per condition, (n_freqs, n_sites) recapture
    params: np.ndarray
    errors: np.ndarray
    min_fit_amplitude: float  # fits with a smaller fringe amplitude are "low contrast"

    @property
    def n_conditions(self):
        return len(self.conditions)

    @property
    def condition_names(self):
        return [", ".join(f"{k} = {v:g}" for k, v in c.items()) for c in self.conditions]

    @property
    def f0(self):
        """Fitted resonance (MHz), shape ``(n_conditions, n_sites)``."""
        return self.params[..., 3]

    @property
    def has_fit(self):
        return np.isfinite(self.f0)

    @property
    def reliable(self):
        """A fit exists and its fringe amplitude is at least ``min_fit_amplitude``."""
        return self.has_fit & (np.nan_to_num(self.params[..., 1]) >= self.min_fit_amplitude)

    def light_shifts(self):
        """``(shift, error)`` in kHz, shape ``(n_conditions, n_arena)``: arena f0 - reservoir f0.

        NaN where the site or the reservoir fit is not reliable.
        """
        ok = self.reliable[:, 1:] & self.reliable[:, :1]
        shift = (self.f0[:, 1:] - self.f0[:, :1]) * 1e3
        error = np.hypot(self.errors[:, 1:, 3], self.errors[:, :1, 3]) * 1e3
        return np.where(ok, shift, np.nan), np.where(ok, error, np.nan)

    def summary(self):
        lines = []
        for i, name in enumerate(self.condition_names):
            lines += [name or self.label,
                      f"{'':10s} {'f0 (MHz)':>20s} {'f_rabi (kHz)':>14s} {'T (us)':>12s} {'A':>6s} {'c':>6s}"]
            for k, site in enumerate(self.site_names):
                if not self.has_fit[i, k]:
                    lines.append(f"{site:10s} no fit")
                    continue
                p, e = self.params[i, k], self.errors[i, k]
                flag = "" if self.reliable[i, k] else "  low contrast"
                lines.append(f"{site:10s} {p[3]:11.4f} ± {e[3]:.4f} {p[2]*1e3:9.1f} ± {e[2]*1e3:3.1f} "
                             f"{p[4]:6.1f} ± {e[4]:3.1f} {p[1]:6.2f} {p[0]:6.2f}{flag}")
            lines.append("")
        return "\n".join(lines).rstrip()


def split_conditions(exp, frequency_axis):
    """``[(condition, 1-axis frequency scan), ...]``, one per combination of the other axes."""
    others = [a for a in range(exp.scan_nb_axes) if a != frequency_axis]
    if not others:
        return [({}, exp)]
    result = []
    for values in np.unique(exp.param_sets_array[:, others], axis=0):
        condition = {" ".join(exp.parameters_labels[a]): float(v) for a, v in zip(others, values)}
        sub = exp
        for axis, value in sorted(zip(others, values), reverse=True):  # highest axis first
            sub = sub.fix_axis(axis, value)
        result.append((condition, sub))
    return result


def fit_arena_scans(exp, *, n_arena=7, frequency_axis=None, condition=None,
                    min_fit_amplitude=0.15, verbose=True, **fit_kwargs):
    """Fit the reservoir and every arena site for every condition of ``exp``.

    ``frequency_axis``: default is the axis whose parameter name contains
    "frequency" (else the last axis). ``condition``: settings that were not
    scanned, e.g. ``{"Arena AOM amplitude": 0.12}``, added to every condition
    for titles and legends. ``fit_kwargs`` go to ``fit_ramsey``.
    """
    if frequency_axis is None:
        names = [str(parameter).lower() for _, parameter in exp.parameters_labels]
        frequency_axis = next((a for a, n in enumerate(names) if "frequency" in n), len(names) - 1)

    conditions, freqs, curves, params, errors = [], [], [], [], []
    for cond, sub in split_conditions(exp, frequency_axis):
        f, c = site_curves(sub, n_arena)
        fits = [fit_ramsey(f, y, **fit_kwargs) for y in c.T]
        conditions.append({**(condition or {}), **cond})
        freqs.append(f)
        curves.append(c)
        params.append([p if p is not None else np.full(5, np.nan) for p, _ in fits])
        errors.append([e if e is not None else np.full(5, np.nan) for _, e in fits])

    fits = ArenaFits(
        label=exp.label,
        frequency_label=" ".join(exp.parameters_labels[frequency_axis]),
        conditions=conditions, site_names=site_names(n_arena), freqs=freqs, curves=curves,
        params=np.array(params), errors=np.array(errors), min_fit_amplitude=min_fit_amplitude,
    )
    if verbose:
        print(fits.summary())
    return fits


# --------------------------------------------------------------------------- #
# Beam center
# --------------------------------------------------------------------------- #

@dataclass
class BeamFit:
    """Gaussian beam ``shift = S_c exp(-2 |r - center|² / waist²)`` fitted to the light shifts.

    The conditions share ``center`` and ``waist``; each has its own peak shift ``S_c``.
    """

    center: np.ndarray        # (x0, y0), in `unit`
    center_err: np.ndarray
    waist: float              # 1/e² intensity radius
    waist_err: float
    peak_shifts: np.ndarray   # (n_conditions,), kHz
    peak_errors: np.ndarray
    conditions: list
    condition_names: list
    site_names: list          # arena sites
    site_xy: np.ndarray       # (n_arena, 2)
    shifts: np.ndarray        # (n_conditions, n_arena), kHz, NaN where unreliable
    shift_errors: np.ndarray
    label: str
    unit: str = "px"         # unit of the positions

    @property
    def arena_center(self):
        return self.site_xy.mean(axis=0)

    @property
    def offset(self):
        """Beam center minus arena center."""
        return self.center - self.arena_center

    @property
    def ring_radius(self):
        """Mean distance of the outer sites to the arena center."""
        d = np.linalg.norm(self.site_xy - self.arena_center, axis=1)
        return d[d > 0.5 * d.max()].mean()

    def shift_model(self, r):
        """Model shift (kHz) at distance ``r``, shape ``(n_conditions, len(r))``."""
        return self.peak_shifts[:, None] * np.exp(-2 * np.asarray(r)[None, :] ** 2 / self.waist ** 2)

    def summary(self):
        (x0, y0), (dx0, dy0), (ax, ay), (ox, oy) = self.center, self.center_err, self.arena_center, self.offset
        distance, u = np.hypot(ox, oy), self.unit
        lines = [
            f"Fit of {np.isfinite(self.shifts).sum()} shifts, {len(self.conditions)} condition(s):",
            f"  beam center   x0 = {x0:.2f} ± {dx0:.2f} {u},  y0 = {y0:.2f} ± {dy0:.2f} {u}",
            f"  arena center  x  = {ax:.2f} {u},        y  = {ay:.2f} {u}",
            f"  offset        dx = {ox:+.2f} {u}, dy = {oy:+.2f} {u}  "
            f"(|d| = {distance:.2f} {u} = {distance / self.ring_radius:.2f} ring radius)",
            f"  waist         w  = {self.waist:.2f} ± {self.waist_err:.2f} {u}",
        ]
        for name, S, dS in zip(self.condition_names, self.peak_shifts, self.peak_errors):
            lines.append(f"  peak shift{' at ' + name if name else ''}: {S:.1f} ± {dS:.1f} kHz")
        return "\n".join(lines)


def fit_beam_center(fits, site_xy, conditions=None, unit="px"):
    """Fit a Gaussian beam to the reliable light shifts.

    ``site_xy``: arena site positions, in ``unit``: trap-file positions in um
    (``trap_file_positions``) or camera ROI px (``arena_positions``). ``conditions``:
    indices of the conditions to include (default all); they share the center,
    so fit conditions that move the beam or the tweezers one at a time,
    e.g. ``[fit_beam_center(fits, xy, [i]) for i in range(fits.n_conditions)]``.
    """
    idx = list(range(fits.n_conditions)) if conditions is None else list(conditions)
    shifts, shift_errors = (a[idx] for a in fits.light_shifts())
    xy = np.asarray(site_xy, dtype=float)
    rows, sites = np.nonzero(np.isfinite(shifts))
    if len(rows) < 3 + len(idx):
        raise ValueError(f"Only {len(rows)} reliable arena shifts for {3 + len(idx)} beam parameters; "
                         "check the Ramsey fits (low contrast, reservoir f0)")

    def model(_, x0, y0, w, *peaks):
        r2 = ((xy[sites] - [x0, y0]) ** 2).sum(axis=1)
        return np.asarray(peaks)[rows] * np.exp(-2 * r2 / w ** 2)

    x_min, y_min = xy.min(axis=0) - 30
    x_max, y_max = xy.max(axis=0) + 30
    spacing = min(np.linalg.norm(p - q) for k, p in enumerate(xy) for q in xy[k + 1:])
    p0 = [*xy.mean(axis=0), 2 * spacing] + [max(np.nanmax(shifts), 1.0)] * len(idx)
    bounds = ([x_min, y_min, 0.2 * spacing] + [0] * len(idx), [x_max, y_max, 40 * spacing] + [np.inf] * len(idx))
    p, cov = curve_fit(model, None, shifts[rows, sites], p0=p0, bounds=bounds, maxfev=20000)
    err = np.sqrt(np.diag(cov))

    return BeamFit(
        center=p[:2], center_err=err[:2], waist=p[2], waist_err=err[2],
        peak_shifts=p[3:], peak_errors=err[3:],
        conditions=[fits.conditions[i] for i in idx],
        condition_names=[fits.condition_names[i] for i in idx],
        site_names=fits.site_names[1:], site_xy=xy, shifts=shifts, shift_errors=shift_errors,
        label=fits.label, unit=unit,
    )


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Whole analysis of one scan
# --------------------------------------------------------------------------- #

# Calibrated SLM trap file: design positions (um) of the traps, in the same
# units as the SLM grating, so a beam offset in these coordinates is the grating step.
TRAP_FILE = r"C:\Users\PETANQUE-PC\Desktop\Codes\SLM-Controller\Tweezer_ROIs\263tweezers_new_calib_3p5.json"


def center_site(site_xy):
    """Index of the arena site closest to the centroid of the arena ROIs."""
    return int(np.argmin(np.linalg.norm(site_xy - site_xy.mean(axis=0), axis=1)))


def ring_inhomogeneity(fits, ring_sites):
    """``(mean shift kHz, std / |mean|, n sites used)`` of the ring, one value per condition.

    Only reliable light shifts are used.
    """
    shifts = fits.light_shifts()[0][:, ring_sites]
    with warnings.catch_warnings():  # conditions without any reliable shift
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(shifts, axis=1)
        return mean, np.nanstd(shifts, axis=1) / np.abs(mean), np.isfinite(shifts).sum(axis=1)


def grating_step(beam):
    """``(step, error)`` in um: SLM grating move that brings the arena onto the beam center.

    ``beam`` must be fitted on trap-file positions (um): the step is the beam offset.
    """
    if beam.unit != "um":
        raise ValueError("grating_step needs a beam fitted on trap-file positions (um)")
    return beam.offset.copy(), beam.center_err.copy()


@dataclass
class ArenaRamseyAnalysis:
    fits: ArenaFits
    beam: BeamFit
    ring_sites: list
    inhomogeneity: np.ndarray   # std / |mean| of the ring shifts, per condition
    step: np.ndarray            # grating move (um)
    step_err: np.ndarray
    next_grating: np.ndarray    # None if the current grating is not given
    fig: plt.Figure


def analyze_arena_ramsey(exp, *, condition=None, grating=None, n_arena=7, min_fit_amplitude=0.15,
                         site_styles=None, trap_file=TRAP_FILE, figsize=(16, 13), show=True, verbose=True):
    """Full arena Ramsey analysis of one condition, in one call.

    1. Ramsey fit of the reservoir average and of each arena site (fit table printed).
    2. Ring inhomogeneity: std / |mean| of the light shift of the 6 ring sites (printed).
    3. Gaussian fit of the beam center, and the SLM grating step towards it.
    4. One figure: all site curves together (top), beam map and shift vs distance (bottom).
    5. After the figure: beam fit summary, step, and the next grating if ``grating`` is given.

    ``grating``: current SLM grating (x, y) in um. ``trap_file``: calibrated SLM trap
    file; its last ``n_arena`` positions (um) are the arena sites for the beam fit.
    Returns an ``ArenaRamseyAnalysis``.
    """
    fits = fit_arena_scans(exp, n_arena=n_arena, condition=condition,
                           min_fit_amplitude=min_fit_amplitude, verbose=verbose)
    site_xy = trap_file_positions(trap_file, n_arena)
    center = center_site(site_xy)
    ring = [i for i in range(n_arena) if i != center]
    mean, inhomogeneity, n_used = ring_inhomogeneity(fits, ring)
    print()
    for name, value, m, n in zip(fits.condition_names, inhomogeneity, mean, n_used):
        print(f"{name}: mean shift = {m:.3f} kHz, std/mean over {n} ring atoms = {value * 100:.1f} %")

    beam = fit_beam_center(fits, site_xy, unit="um")
    step, step_err = grating_step(beam)
    next_grating = None if grating is None else np.asarray(grating, dtype=float) + step

    fig = plt.figure(figsize=figsize, layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.1])
    map_fig, dist_fig = bottom.subfigures(1, 2, width_ratios=[1.2, 1])
    ax = plot_arena_fits_overlay(fits, 0, ax=top.subplots(), site_styles=site_styles)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plot_beam_map(beam, fig=map_fig)
    map_fig.suptitle("+ arena center, ★ beam center, dashed circle = w\n"
                     "arrow = move from beam center to arena center", fontsize="medium")
    for ax in map_fig.axes[:len(beam.conditions)]:  # site panels (colorbars come after)
        ax.grid(False)
        ax.minorticks_off()
    ax = plot_shift_vs_distance(beam, ax=dist_fig.subplots())
    ax.legend(loc="upper right")
    if show:
        plt.show()

    print()
    print(beam.summary())
    print(f"\nStep                   = ({step[0]:+.2f} ± {step_err[0]:.2f}, {step[1]:+.2f} ± {step_err[1]:.2f}) um"
          f"  (|step| = {np.linalg.norm(step):.2f} um)")
    if next_grating is not None:
        print(f"Current grating (x, y) = ({grating[0]:.2f}, {grating[1]:.2f}) um")
        print(f"Next grating to try    = ({next_grating[0]:.2f}, {next_grating[1]:.2f}) um")
    return ArenaRamseyAnalysis(fits, beam, ring, inhomogeneity, step, step_err, next_grating, fig)
