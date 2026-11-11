"""Step 2: Gaussian beam fit of the arena light shifts, and the SLM grating step.

The light shift is proportional to the arena beam intensity, so the shifts of
the arena sites follow its Gaussian profile::

    shift(r) = S * exp(-2 |r - center|² / w²)      w: 1/e² intensity radius

Site positions are the design positions (um) of the calibrated SLM trap file,
the same coordinates as the SLM grating: the offset of the beam center from the
arena center is directly the grating step.
"""

import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

# Calibrated SLM trap file: design positions (um) of the traps.
TRAP_FILE = r"C:\Users\PETANQUE-PC\Desktop\Codes\SLM-Controller\Tweezer_ROIs\263tweezers_new_calib_3p5.json"


def trap_file_positions(path, n_arena=7):
    """Design positions (um) of the arena sites, shape ``(n_arena, 2)``, from an SLM trap file.

    The last ``n_arena`` traps of the file, in the same order as the arena ROIs.
    """
    with open(path, encoding="utf-8") as f:
        return np.array(json.load(f)["centers"], dtype=float)[-n_arena:]


def center_site(site_xy_um):
    """Index of the arena site closest to the centroid of the arena sites."""
    return int(np.argmin(np.linalg.norm(site_xy_um - site_xy_um.mean(axis=0), axis=1)))


def min_site_spacing(site_xy_um):
    """Smallest distance (um) between two arena sites."""
    xy = np.asarray(site_xy_um, dtype=float)
    return min(np.linalg.norm(p - q) for k, p in enumerate(xy) for q in xy[k + 1:])


def ring_inhomogeneity(fits, ring_sites):
    """``(mean shift kHz, std / |mean|, number of sites used)`` of the reliable ring shifts.

    ``ring_sites``: indices of the ring sites among the arena sites. NaN (and 0
    sites) when no ring shift is reliable.
    """
    shifts = fits.light_shifts_kHz()[0][ring_sites]
    used = shifts[np.isfinite(shifts)]
    if used.size == 0:
        return np.nan, np.nan, 0
    mean = used.mean()
    return mean, used.std() / abs(mean), used.size


@dataclass
class BeamFit:
    """Gaussian beam ``shift = S exp(-2 |r - center|² / w²)`` fitted to the light shifts of one condition."""

    center_um: np.ndarray        # (x0, y0)
    center_err_um: np.ndarray
    waist_um: float              # w, 1/e² intensity radius
    waist_err_um: float
    peak_shift_kHz: float        # S, shift at the beam center
    peak_shift_err_kHz: float
    site_names: list             # arena sites
    site_xy_um: np.ndarray       # (n_arena, 2)
    shifts_kHz: np.ndarray       # (n_arena,), NaN where not reliable (not used in the fit)
    shift_errors_kHz: np.ndarray
    condition_name: str
    label: str

    @property
    def arena_center_um(self):
        return self.site_xy_um.mean(axis=0)

    @property
    def offset_um(self):
        """Beam center minus arena center."""
        return self.center_um - self.arena_center_um

    @property
    def ring_radius_um(self):
        """Mean distance of the outer sites to the arena center."""
        d = np.linalg.norm(self.site_xy_um - self.arena_center_um, axis=1)
        return d[d > 0.5 * d.max()].mean()  # the center site is at d ~ 0

    def shift_model_kHz(self, r_um):
        """Model shift (kHz) at distance ``r_um`` from the beam center."""
        return self.peak_shift_kHz * np.exp(-2 * np.asarray(r_um) ** 2 / self.waist_um ** 2)

    def summary(self):
        (x0, y0), (dx0, dy0) = self.center_um, self.center_err_um
        (ax, ay), (ox, oy) = self.arena_center_um, self.offset_um
        distance = np.hypot(ox, oy)
        condition = f" at {self.condition_name}" if self.condition_name else ""
        return "\n".join([
            f"Fit of {np.isfinite(self.shifts_kHz).sum()} shifts:",
            f"  beam center   x0 = {x0:.2f} ± {dx0:.2f} um,  y0 = {y0:.2f} ± {dy0:.2f} um",
            f"  arena center  x  = {ax:.2f} um,        y  = {ay:.2f} um",
            f"  offset        dx = {ox:+.2f} um, dy = {oy:+.2f} um  "
            f"(|d| = {distance:.2f} um = {distance / self.ring_radius_um:.2f} ring radius)",
            f"  waist         w  = {self.waist_um:.2f} ± {self.waist_err_um:.2f} um",
            f"  peak shift{condition}: {self.peak_shift_kHz:.1f} ± {self.peak_shift_err_kHz:.1f} kHz",
        ])


def gaussian_beam_kHz(xy_um, x0_um, y0_um, waist_um, peak_shift_kHz):
    """Light shift (kHz) at the positions ``xy_um`` (shape ``(n, 2)``) of a Gaussian beam."""
    r2 = ((xy_um - [x0_um, y0_um]) ** 2).sum(axis=1)
    return peak_shift_kHz * np.exp(-2 * r2 / waist_um ** 2)


def fit_beam_center(fits, site_xy_um, *, search_margin_um=30):
    """Fit a Gaussian beam to the reliable light shifts of one condition.

    Args:
        fits: ``ArenaFits`` of one condition.
        site_xy_um: Arena site positions (um), from ``trap_file_positions``.
        search_margin_um: The beam center is searched within this distance of the sites.

    Returns:
        A ``BeamFit``.
    """
    shifts, shift_errors = fits.light_shifts_kHz()
    xy = np.asarray(site_xy_um, dtype=float)
    used = np.isfinite(shifts)
    if used.sum() < 4:
        raise ValueError(f"Only {used.sum()} reliable arena shifts for 4 beam parameters; "
                         "check the Ramsey fits (low contrast, reservoir f0)")

    spacing = min_site_spacing(xy)
    #     x0, y0                 waist         peak shift
    p0 = [*xy.mean(axis=0),      2 * spacing,  max(np.nanmax(shifts), 1.0)]
    lower = [*(xy.min(axis=0) - search_margin_um), 0.2 * spacing, 0]
    upper = [*(xy.max(axis=0) + search_margin_um), 40 * spacing,  np.inf]
    p, cov = curve_fit(gaussian_beam_kHz, xy[used], shifts[used], p0=p0, bounds=(lower, upper),
                       maxfev=20000)
    err = np.sqrt(np.diag(cov))

    return BeamFit(
        center_um=p[:2], center_err_um=err[:2], waist_um=p[2], waist_err_um=err[2],
        peak_shift_kHz=p[3], peak_shift_err_kHz=err[3],
        site_names=[site.name for site in fits.arena], site_xy_um=xy,
        shifts_kHz=shifts, shift_errors_kHz=shift_errors,
        condition_name=fits.condition_name, label=fits.label,
    )


def grating_step(beam):
    """``(step, error)`` in um: the SLM grating move that brings the arena center onto the beam center."""
    return beam.offset_um.copy(), beam.center_err_um.copy()
