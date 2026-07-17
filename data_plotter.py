from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .plotlive_loader import ExperimentData
import matplotlib.pyplot as plt
import numpy as np

class DataPlotter():

    def plot_recapture(self, exp : ExperimentData, threshold=212):
        """
        Plot the recapture probability for an experiment as a function of the scanned parameter. (only works for 1D scan)

        Parameters
        ----------
        self : DataPlotter
            The instance of the DataPlotter class.
        exp : ExperimentData
            The experimental data object containing data arrays and labels.
        threshold : int, optional
            The fluorescence threshold used to compute the recapture array (default is 212).

        This method plots the recapture probability versus the first parameter in the parameter set array,
        masking any values where the recapture probability is None.
        """
        recapture = exp.compute_recapture_array(threshold)
        mask = np.array([value is not None for value in recapture], dtype=bool)

        plt.plot(
            exp.param_sets_array[mask, 0],
            np.array(recapture[mask], dtype=float),
            marker='o',
            linestyle='-',
            label=exp.folder_name
        )
        plt.title("Recapture")
        plt.ylabel("Recapture Probability")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc=(1.02, 0.0))
        plt.ylim(-0.03, 1.03)
        plt.xlabel(exp.parameters_labels[0])
   

    def plot_loading(self, exp : ExperimentData, threshold=212):
        loading = exp.compute_loading_array(threshold=threshold)
        mask = np.array([value is not None for value in loading], dtype=bool)

        plt.plot(
            exp.param_sets_array[mask, 0],
            np.array(loading[mask], dtype=float),
            marker='o',
            linestyle='-',
            label=exp.folder_name
        )
        plt.title("Loading")
        plt.ylabel("Loading Probability")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc=(1.02, 0.0))
        plt.ylim(-0.03, 1.03)
        plt.xlabel(exp.parameters_labels[0])

    def plot_fluorescence(self, exp : ExperimentData, image_index, threshold=212):
        data = exp.data_array[:, :, image_index, :].astype(float)

        # Separate data points into those above and below the fluorescence threshold
        above = np.where(data > threshold, data, np.nan)
        below = np.where(data < threshold, data, np.nan)
 
        fluorescence_above = np.nanmean(above, axis=(1, 2))
        fluorescence_below = np.nanmean(below, axis=(1, 2))

        err_above = np.nanstd(above, axis=(1, 2))
        err_below = np.nanstd(below, axis=(1, 2))

        mask_above = ~np.isnan(fluorescence_above)
        mask_below = ~np.isnan(fluorescence_below)

        x_above = exp.param_sets_array[mask_above, 0]
        y_above = fluorescence_above[mask_above]
        yerr_above = err_above[mask_above]
        x_below = exp.param_sets_array[mask_below, 0]
        y_below = fluorescence_below[mask_below]
        yerr_below = err_below[mask_below]
        xlabel = exp.parameters_labels[0]

        fig = plt.figure(figsize=(14, 4))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
        fig.suptitle(f"Fluorescence Image #{image_index + 1}")
        ax_both = fig.add_subplot(gs[:, 0])
        ax_above = fig.add_subplot(gs[0, 1])
        ax_below = fig.add_subplot(gs[1, 1], sharex=ax_above)

        err_kw = dict(fmt='o-', capsize=3, elinewidth=1)

        # Combined plot (left)
        ax_both.errorbar(x_above, y_above, yerr=yerr_above, label="above threshold", **err_kw)
        ax_both.errorbar(x_below, y_below, yerr=yerr_below, label="below threshold", **err_kw)
        ax_both.axhline(threshold, color='gray', linestyle='--', alpha=0.6, label=f"Threshold = {threshold}")

        ax_both.set_ylabel("Counts")
        ax_both.set_xlabel(xlabel)
        ax_both.grid(True, linestyle='--', alpha=0.6)
        ax_both.legend(loc='best')

        # Above / below (right, stacked)
        ax_above.errorbar(x_above, y_above, yerr=yerr_above, label=exp.folder_name, **err_kw)
        ax_above.grid(True, linestyle='--', alpha=0.6)

        ax_below.errorbar(
            x_below, y_below, yerr=yerr_below, label=exp.folder_name,
            color=plt.cm.tab20.colors[3], markeredgecolor=plt.cm.tab20.colors[2], **err_kw,
        )
        ax_below.set_xlabel(xlabel)
        ax_below.grid(True, linestyle='--', alpha=0.6)
