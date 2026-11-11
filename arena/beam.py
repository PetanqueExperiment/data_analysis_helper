"""Gaussian beam fit of the arena light shifts and SLM grating step (moved from arena_light_shift.py)."""

import json
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

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
