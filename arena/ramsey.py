"""Step 1: Ramsey fit of the reservoir and of each arena site, and their light shifts.

Sites: the last ``n_arena`` ROIs are the arena sites (a ring of 6 plus its
center); all other ROIs form the reservoir, which sees no arena light. The
reservoir is fitted on its mean recapture and gives the reference resonance,
so the light shift of an arena site is ``f0(site) - f0(reservoir)``.

Conditions: the scan has one microwave frequency axis and any number of other
axes (arena amplitude, grating, repetition, ...). Each combination of the
other axes is a *condition*; ``fit_arena_scans`` returns one ``ArenaFits`` per
condition.

Units: frequency MHz, Ramsey time T μs, light shift kHz.
"""

import warnings
from dataclasses import dataclass

from ..experiment import ExperimentData
import numpy as np
from scipy.optimize import curve_fit

from ..models import ramsey_spectro

RESERVOIR = "Reservoir"


def ramsey_no_phase(f_MHz, offset, amplitude, f_rabi_MHz, f0_MHz, T_us):
    """Ramsey lineshape (``models.ramsey_spectro``) with the phase fixed to 0."""
    return ramsey_spectro(f_MHz, offset, amplitude, f_rabi_MHz, f0_MHz, T_us, phi=0)


def fit_ramsey(f_MHz, recapture, T_max_us=25, f_rabi_starts_MHz=(0.01, 0.02), n_f0_starts=10,
               min_contrast=0.05):
    """Fit ``ramsey_no_phase`` from a grid of starting points; keep the best fit.

    Args:
        f_MHz: Microwave frequencies.
        recapture: Recapture probability at each frequency; NaN points are skipped.
        T_max_us: Upper bound of the fitted Ramsey time, so the fit cannot jump
            to an aliased fringe.
        f_rabi_starts_MHz: Starting values of the Rabi frequency.
        n_f0_starts: Number of starting resonances, spread over the scan.
        min_contrast: Curves with a smaller peak-to-peak are not fitted (flat).

    Returns:
        ``(params, errors)`` with params ``(offset, amplitude, f_rabi_MHz,
        f0_MHz, T_us)``, or ``(None, None)`` for a flat curve, fewer than 6
        points, or no convergence.
    """
    ok = np.isfinite(recapture)
    f, y = f_MHz[ok], recapture[ok]
    if len(f) < 6 or np.ptp(y) < min_contrast:
        return None, None
    #        offset  amplitude  f_rabi_MHz  f0_MHz   T_us
    lower = [-0.1,   0,         1e-3,       f.min(), 0]
    upper = [1,      2,         0.5,        f.max(), T_max_us]
    margin = 0.2 * np.ptp(f)  # starting resonances stay away from the scan edges
    best_residual, best = np.inf, (None, None)
    for f_rabi in f_rabi_starts_MHz:
        for f0 in np.linspace(f.min() + margin, f.max() - margin, n_f0_starts):
            p0 = [y.min(), np.ptp(y), f_rabi, f0, T_max_us / 2]
            try:
                p, cov = curve_fit(ramsey_no_phase, f, y, p0=p0, bounds=(lower, upper), maxfev=5000)
            except (RuntimeError, ValueError):
                continue
            residual = np.sum((ramsey_no_phase(f, *p) - y) ** 2)
            if residual < best_residual:
                best_residual, best = residual, (p, np.sqrt(np.diag(cov)))
    return best


@dataclass
class RamseyFit:
    """Ramsey fit of one site (or of the reservoir average).

    ``params`` / ``errors``: ``(offset, amplitude, f_rabi_MHz, f0_MHz, T_us)``,
    all NaN when there is no fit.
    """

    name: str
    recapture: np.ndarray   # (n_freqs,), at ArenaFits.freqs_MHz
    params: np.ndarray
    errors: np.ndarray
    reliable: bool          # a fit exists and its fringe amplitude is >= min_fit_amplitude

    @property
    def has_fit(self):
        return bool(np.isfinite(self.params[3]))

    @property
    def f0_MHz(self):
        return self.params[3]

    @property
    def f0_err_MHz(self):
        return self.errors[3]


@dataclass
class ArenaFits:
    """Ramsey fits of the reservoir and of every arena site, for one condition."""

    label: str                # experiment folder
    frequency_label: str
    condition: dict           # {"device parameter": value} of this condition
    freqs_MHz: np.ndarray     # sorted
    reservoir: RamseyFit
    arena: list               # one RamseyFit per arena site
    min_fit_amplitude: float  # fits with a smaller fringe amplitude are "low contrast"

    @property
    def sites(self):
        """Reservoir first, then the arena sites (the order of ``site_styles``)."""
        return [self.reservoir, *self.arena]

    @property
    def condition_name(self):
        return ", ".join(f"{k} = {v:g}" for k, v in self.condition.items())

    def light_shifts_kHz(self):
        """``(shift, error)`` of each arena site, in kHz: f0(site) - f0(reservoir).

        NaN where the site or the reservoir fit is not reliable. The two f0 come
        from independent fits, so their errors add in quadrature.
        """
        reference = self.reservoir
        shifts, errors = [], []
        for site in self.arena:
            if site.reliable and reference.reliable:
                shifts.append((site.f0_MHz - reference.f0_MHz) * 1e3)  # MHz -> kHz
                errors.append(np.hypot(site.f0_err_MHz, reference.f0_err_MHz) * 1e3)
            else:
                shifts.append(np.nan)
                errors.append(np.nan)
        return np.array(shifts), np.array(errors)

    def summary(self):
        lines = [self.condition_name or self.label,
                 f"{'':10s} {'f0 (MHz)':>20s} {'f_rabi (kHz)':>14s} {'T (us)':>12s} {'A':>6s} {'c':>6s}"]
        for site in self.sites:
            if not site.has_fit:
                lines.append(f"{site.name:10s} no fit")
                continue
            (c, A, f_rabi, f0, T), (_, _, df_rabi, df0, dT) = site.params, site.errors
            flag = "" if site.reliable else "  low contrast"
            lines.append(f"{site.name:10s} {f0:11.4f} ± {df0:.4f} "
                         f"{f_rabi * 1e3:9.1f} ± {df_rabi * 1e3:3.1f} "
                         f"{T:6.1f} ± {dT:3.1f} {A:6.2f} {c:6.2f}{flag}")
        return "\n".join(lines)


def site_curves(sub_exp, n_arena):
    """``(freqs_MHz, reservoir, arena)`` of a 1-axis frequency scan, sorted by frequency.

    ``reservoir``: mean recapture over the reservoir sites, shape ``(n_freqs,)``.
    ``arena``: recapture of each arena site, shape ``(n_freqs, n_arena)``.
    """
    recapture = sub_exp.compute_recapture_per_atom()
    with warnings.catch_warnings():  # reservoir sites never loaded are NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        reservoir = np.nanmean(recapture[:, :-n_arena], axis=1)
    freqs = sub_exp.param_sets_array[:, 0]
    order = np.argsort(freqs)
    return freqs[order], reservoir[order], recapture[order, -n_arena:]


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


def _fit_site(name, freqs_MHz, recapture, min_fit_amplitude, fit_kwargs):
    ramsey_params, errors = fit_ramsey(freqs_MHz, recapture, **fit_kwargs)
    if ramsey_params is None:
        ramsey_params, errors = np.full(5, np.nan), np.full(5, np.nan)
    reliable = bool(np.isfinite(ramsey_params[3]) and ramsey_params[1] >= min_fit_amplitude)
    return RamseyFit(name, recapture, ramsey_params, errors, reliable)


def fit_arena_scans(exp : ExperimentData, *, n_arena=7, frequency_axis=None, condition=None,
                    min_fit_amplitude=0.15, verbose=True, **fit_kwargs):
    """Fit the reservoir and every arena site, for every condition of ``exp``.

    Args:
        exp: ``ExperimentData`` with one frequency axis (MHz).
        n_arena: The last ``n_arena`` ROIs are the arena sites.
        frequency_axis: Default: the axis whose parameter name contains
            "frequency", else the last axis.
        condition: Settings that were not scanned, e.g.
            ``{"Arena AOM amplitude": 0.12}``, added to every condition for
            titles and legends.
        min_fit_amplitude: Fits with a smaller fringe amplitude are not reliable
            and give no light shift.
        verbose: Print the fit table of each condition.
        **fit_kwargs: Passed to ``fit_ramsey``.

    Returns:
        A list with one ``ArenaFits`` per condition (one element for a
        frequency-only scan).
    """
    if frequency_axis is None:
        names = [str(parameter).lower() for _, parameter in exp.parameters_labels]
        frequency_axis = next((a for a, n in enumerate(names) if "frequency" in n), len(names) - 1)

    result = []
    for scanned, sub in split_conditions(exp, frequency_axis):
        freqs, reservoir_curve, arena_curves = site_curves(sub, n_arena)
        reservoir = _fit_site(RESERVOIR, freqs, reservoir_curve, min_fit_amplitude, fit_kwargs)
        arena = [_fit_site(f"Arena {i}", freqs, arena_curves[:, i], min_fit_amplitude, fit_kwargs)
                 for i in range(n_arena)]
        fits = ArenaFits(
            label=exp.label,
            frequency_label=" ".join(exp.parameters_labels[frequency_axis]),
            condition={**(condition or {}), **scanned},
            freqs_MHz=freqs, reservoir=reservoir, arena=arena, min_fit_amplitude=min_fit_amplitude,
        )
        if verbose:
            print(fits.summary(), end="\n\n")
        result.append(fits)
    return result
