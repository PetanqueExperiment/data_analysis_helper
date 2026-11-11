"""Whole arena Ramsey analysis of one scan (moved from arena_light_shift.py)."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from .beam import TRAP_FILE, BeamFit, center_site, fit_beam_center, grating_step, ring_inhomogeneity, trap_file_positions
from .plots import plot_arena_fits_overlay, plot_beam_map, plot_shift_vs_distance
from .ramsey import ArenaFits, fit_arena_scans


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
