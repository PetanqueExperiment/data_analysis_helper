"""Arena light-shift analysis from Ramsey scans of arena vs reservoir sites.

Read the files in the order of the analysis:

``ramsey.py``    recapture of each site -> Ramsey fit -> light shift (kHz)
``beam.py``      light shifts of the arena sites -> Gaussian beam center -> SLM grating step (um)
``analysis.py``  the whole chain for one scan, and its printed / plotted report
``plots.py``     the figures (check them by eye)

Typical use::

    from data_analysis_helper import arena as als

    result = als.analyze_arena_ramsey(exp, grating=(41.38, 51.18))  # all steps, one figure

    # or step by step, for every condition of a scan
    for fits in als.fit_arena_scans(exp, n_arena=7):
        als.plot_arena_fits_overlay(fits)
        beam = als.fit_beam_center(fits, als.trap_file_positions(als.TRAP_FILE))
        print(beam.summary())
        als.plot_beam_map(beam)
"""

from .analysis import ArenaRamseyAnalysis, analyze_arena_ramsey, compute_arena_ramsey, report_arena_ramsey
from .beam import (TRAP_FILE, BeamFit, center_site, fit_beam_center, gaussian_beam_kHz, grating_step,
                   min_site_spacing, ring_inhomogeneity, trap_file_positions)
from .plots import (default_styles, plot_arena_fits, plot_arena_fits_overlay, plot_beam_map,
                    plot_resonance_vs_condition, plot_shift_vs_distance)
from .ramsey import RESERVOIR, ArenaFits, RamseyFit, fit_arena_scans, fit_ramsey, ramsey_no_phase
