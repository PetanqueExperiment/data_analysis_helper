"""Analysis helpers for PyCam tweezer experiments.

Modules
-------
experiment
    :class:`ExperimentData` — load an experiment folder and compute loading /
    recapture probabilities.
data_plotter
    ``plot_*`` functions taking an ``ExperimentData``; styling is left to the
    caller.
models
    Fit models (``rabi_osc``, ``ramsey_spectro``) for ``curve_fit``.
release_recapture
    Release–recapture simulation and temperature fit.

The main names are re-exported here::

    from data_analysis_helper import ExperimentData, plot_recapture, rabi_osc
"""

from .data_plotter import (
    plot_fluo_recapture,
    plot_fluorescence,
    plot_loading,
    plot_recapture,
    plot_recapture_2d,
    plot_recapture_per_atom,
)
from .experiment import ExperimentData
from .models import rabi_osc, ramsey_spectro
from .release_recapture import (
    ReleaseRecaptureFit,
    fit_release_recapture,
    plot_release_recapture_fit,
    release_recapture,
)

__all__ = [
    "ExperimentData",
    "plot_recapture",
    "plot_loading",
    "plot_fluorescence",
    "plot_fluo_recapture",
    "plot_recapture_per_atom",
    "plot_recapture_2d",
    "rabi_osc",
    "ramsey_spectro",
    "ReleaseRecaptureFit",
    "fit_release_recapture",
    "plot_release_recapture_fit",
    "release_recapture",
]
