"""Ramsey fits of the reservoir and of every arena site (moved from arena_light_shift.py)."""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from ..models import ramsey_spectro

RESERVOIR = "Reservoir"


def site_names(n_arena):
    """``["Reservoir", "Arena 0", ..., "Arena n-1"]``."""
    return [RESERVOIR] + [f"Arena {i}" for i in range(n_arena)]


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
