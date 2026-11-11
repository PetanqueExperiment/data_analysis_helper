"""Arena light-shift analysis from a Ramsey scan of arena vs reservoir sites.

The last ``n_arena`` ROIs are the arena sites (a ring of 6 plus its center);
all other ROIs form the reservoir, which sees no arena light and gives the
reference resonance.

The scan has one microwave frequency axis and any number of other axes
(arena amplitude, grating, repetition, ...). Each combination of the other
axes is a *condition*, stored as a dict ``{"device parameter": value}``.

Typical use::

    from data_analysis_helper import arena as als

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

from .analysis import ArenaRamseyAnalysis, analyze_arena_ramsey
from .beam import (TRAP_FILE, BeamFit, arena_positions, center_site, fit_beam_center, grating_step,
                   ring_inhomogeneity, trap_file_positions)
from .plots import (default_styles, plot_arena_fits, plot_arena_fits_overlay, plot_beam_map,
                    plot_resonance_vs_condition, plot_shift_vs_distance, plot_site_curves)
from .ramsey import RESERVOIR, ArenaFits, fit_arena_scans, fit_ramsey, ramsey_no_phase, site_curves, site_names, split_conditions
