"""Step 3: the whole arena Ramsey analysis of one frequency-only scan.

``compute_arena_ramsey`` does the calculation (no printing, no plotting):

1. Ramsey fit of the reservoir average and of each arena site -> light shifts.
2. Ring inhomogeneity: std / |mean| of the light shift of the 6 ring sites.
3. Gaussian fit of the beam center on the trap-file positions (um).
4. SLM grating step towards the beam center, and the next grating.

``report_arena_ramsey`` prints the numbers and draws one figure;
``analyze_arena_ramsey`` does both.
"""

import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from .beam import TRAP_FILE, BeamFit, center_site, fit_beam_center, grating_step, ring_inhomogeneity, trap_file_positions
from .plots import plot_arena_fits_overlay, plot_beam_map, plot_shift_vs_distance
from .ramsey import ArenaFits, fit_arena_scans

SLM_LABEL = "SLM_Controller"  # device label of the SLM in the sequencer's static settings


def scan_grating_um(exp):
    """SLM grating (x, y) in um set during the scan, from its ``sequence.json``; None if not recorded."""
    try:
        slm = exp.read_static_settings()[SLM_LABEL]
        return np.array([slm["grating_x"], slm["grating_y"]], dtype=float)
    except (FileNotFoundError, KeyError):
        return None


@dataclass
class ArenaRamseyAnalysis:
    fits: ArenaFits
    beam: BeamFit
    ring_sites: list              # indices of the ring sites among the arena sites
    ring_mean_shift_kHz: float
    inhomogeneity: float          # std / |mean| of the reliable ring shifts
    n_ring_used: int              # number of reliable ring shifts
    step_um: np.ndarray           # grating move (x, y)
    step_err_um: np.ndarray
    grating_um: np.ndarray        # current grating, None if not given
    next_grating_um: np.ndarray   # grating + step, None if the grating is not given
    fig: plt.Figure = None        # set by analyze_arena_ramsey


def compute_arena_ramsey(exp, *, condition=None, grating=None, n_arena=7, min_fit_amplitude=0.15,
                         trap_file=TRAP_FILE):
    """Fits, ring inhomogeneity, beam center and grating step of one frequency-only scan.

    Args:
        exp: ``ExperimentData`` scanning only the microwave frequency (MHz).
        condition: Settings that were not scanned, for titles, e.g.
            ``{"Arena AOM amplitude": 0.05}``.
        grating: SLM grating (x, y) in um used for the scan, to compute the
            next one. Default: read from the scan's ``sequence.json``. A given
            grating that differs from the recorded one is used, with a warning.
        n_arena: The last ``n_arena`` ROIs (and traps of ``trap_file``) are the arena sites.
        min_fit_amplitude: See ``fit_arena_scans``.
        trap_file: Calibrated SLM trap file; its last ``n_arena`` positions (um)
            are the arena sites for the beam fit.

    Returns:
        An ``ArenaRamseyAnalysis`` (without figure).
    """
    fits_per_condition = fit_arena_scans(exp, n_arena=n_arena, condition=condition,
                                         min_fit_amplitude=min_fit_amplitude, verbose=False)
    if len(fits_per_condition) != 1:
        raise ValueError(f"The scan has {len(fits_per_condition)} conditions; this analysis takes a "
                         "frequency-only scan. Use fit_arena_scans and fit_beam_center per condition.")
    fits = fits_per_condition[0]

    site_xy_um = trap_file_positions(trap_file, n_arena)
    ring = [i for i in range(n_arena) if i != center_site(site_xy_um)]
    ring_mean, inhomogeneity, n_ring_used = ring_inhomogeneity(fits, ring)

    beam = fit_beam_center(fits, site_xy_um)
    step, step_err = grating_step(beam)
    recorded = scan_grating_um(exp)
    if grating is None:
        grating = recorded
    else:
        grating = np.asarray(grating, dtype=float)
        if recorded is not None and not np.allclose(grating, recorded):
            warnings.warn(f"Given grating ({grating[0]:.2f}, {grating[1]:.2f}) um differs from the one "
                          f"recorded in sequence.json ({recorded[0]:.2f}, {recorded[1]:.2f}) um; "
                          "using the given one.")
    next_grating = None if grating is None else grating + step
    return ArenaRamseyAnalysis(fits, beam, ring, ring_mean, inhomogeneity, n_ring_used,
                               step, step_err, grating, next_grating)


def report_arena_ramsey(result, *, site_styles=None, figsize=(16, 13), show=True, verbose=True):
    """Print the fit table and ring inhomogeneity, draw one figure, then print the beam fit and step.

    Figure: all site curves with their fits (top), beam map and shift vs
    distance (bottom). Returns the figure.
    """
    fits, beam = result.fits, result.beam
    if verbose:
        print(fits.summary())
        print(f"\n{fits.condition_name}: mean shift = {result.ring_mean_shift_kHz:.3f} kHz, "
              f"std/mean over {result.n_ring_used} ring atoms = {result.inhomogeneity * 100:.1f} %")

    fig = plt.figure(figsize=figsize, layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.1])
    map_fig, dist_fig = bottom.subfigures(1, 2, width_ratios=[1.2, 1])
    ax = plot_arena_fits_overlay(fits, ax=top.subplots(), site_styles=site_styles)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax = plot_beam_map(beam, ax=map_fig.subplots())
    ax.grid(False)
    ax.minorticks_off()
    map_fig.suptitle("+ arena center, ★ beam center, dashed circle = w\n"
                     "arrow = move from beam center to arena center", fontsize="medium")
    ax = plot_shift_vs_distance(beam, ax=dist_fig.subplots())
    if beam.condition_name:  # the only legend entry
        ax.legend(loc="upper right")
    if show:
        plt.show()

    if verbose:
        step, err = result.step_um, result.step_err_um
        print()
        print(beam.summary())
        print(f"\nStep                   = ({step[0]:+.2f} ± {err[0]:.2f}, {step[1]:+.2f} ± {err[1]:.2f}) um"
              f"  (|step| = {np.linalg.norm(step):.2f} um)")
        if result.grating_um is not None:
            (gx, gy), (nx, ny) = result.grating_um, result.next_grating_um
            print(f"Current grating (x, y) = ({gx:.2f}, {gy:.2f}) um")
            print(f"Next grating to try    = ({nx:.2f}, {ny:.2f}) um")
    return fig


def analyze_arena_ramsey(exp, *, condition=None, grating=None, n_arena=7, min_fit_amplitude=0.15,
                         site_styles=None, trap_file=TRAP_FILE, figsize=(16, 13), show=True, verbose=True):
    """``compute_arena_ramsey`` then ``report_arena_ramsey``, in one call.

    Returns the ``ArenaRamseyAnalysis``, with its figure in ``.fig``.
    """
    result = compute_arena_ramsey(exp, condition=condition, grating=grating, n_arena=n_arena,
                                  min_fit_amplitude=min_fit_amplitude, trap_file=trap_file)
    result.fig = report_arena_ramsey(result, site_styles=site_styles, figsize=figsize, show=show,
                                     verbose=verbose)
    return result
