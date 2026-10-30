"""Helpers for loading and inspecting PyCam experiment fluorescence data.

An experiment folder is expected to contain:

- ``manifest.jsonl`` — one JSON object per line; each entry has a ``tag`` (scan
  parameters) and a ``data_file`` path under ``data/``.
- ``rois.json`` — ROI centers (one per trapped atom / tweezers site).
- ``data/`` — ``.npy`` arrays referenced by the manifest.

Manifest tags encode scanned parameters as semicolon-separated tokens::

    device+parameter+value[;device+parameter+value;...]

Example: ``"AOM_Tweezers+amplitude+0.43"`` or a multi-axis scan with several
``;``-separated tokens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

from .data_plotter import *

DATA_SUBDIR = "data"  # subdirectory holding .npy files
MANIFEST_NAME = "manifest.jsonl"
ROIS_NAME = "rois.json"
PARAM_TAG_SEPARATOR = ";"  # separates scan axes in a manifest tag
DEFAULT_THRESHOLD = 217  # fluorescence counts separating "atom" from "no atom"


class ExperimentData(DataPlotter):
    """Load metadata and fluorescence arrays from a single PyCam experiment folder.

    On construction, resolves the folder path, reads the manifest and ROIs,
    and builds two main arrays:

    - ``param_sets_array`` — shape ``(n_entries, n_scan_params)``; string values
      parsed from each manifest ``tag``.
    - ``data_array`` — shape ``(n_entries, *data_shape_per_param_set)``; stacked
      ``.npy`` payloads from ``data/``, NaN-padded for missing shots / files.

    Attributes
    ----------
    folder_name : str | PathLike
        Name or path passed to the constructor.
    base_dir : str | PathLike | None
        Optional parent directory when ``folder_name`` is a relative folder name.
    experiment_folder_path : Path
        Resolved absolute path to the experiment directory.
    manifest : np.ndarray
        Object array of manifest entry dicts (one per line of ``manifest.jsonl``).
    atom_number : int
        Number of ROI centers in ``rois.json``.
    scan_nb_axes : int
        Number of semicolon-separated parameters in the first manifest tag.
    image_nb_per_shot : int
        Images recorded per shot (signal + background); fixed at 2.
    parameters_labels : np.ndarray
        Shape ``(scan_nb_axes, 2)`` — ``(device, parameter)`` for each scan axis.
    total_nb_of_sets_of_params : int
        Number of manifest entries (= number of loaded ``.npy`` files).
    data_shape_per_param_set : tuple
        ``data_array.shape[1:]``, i.e. ``(max_shots, image_nb_per_shot, atom_number)``.
    param_sets_array, data_array : np.ndarray
        Scan values and fluorescence data; see ``read_data``.
    threshold : int
        Default fluorescence threshold used by the compute and plot methods
        when no ``threshold`` argument is passed to them.
    """

    def __init__(
        self,
        folder_name: Union[str, os.PathLike],
        *,
        base_dir: Optional[Union[str, os.PathLike]] = None,
        threshold: int = DEFAULT_THRESHOLD,
        print_summary: bool = True,
    ) -> None:
        """Load an experiment folder and populate all attributes.

        Parameters
        ----------
        folder_name
            Absolute path, path relative to this module's directory, or a folder
            name under ``base_dir``.
        base_dir
            If given, ``folder_name`` is resolved as ``base_dir / folder_name``.
        threshold
            Default fluorescence threshold for this experiment.
        """
        self.folder_name = folder_name
        self.base_dir = base_dir
        self.threshold = threshold
        self.experiment_folder_path = self._resolve_experiment_folder(
            folder_name, base_dir=base_dir
        )
        self.manifest = self.read_manifest()
        self.atom_number = self.read_atom_number()
        self.scan_nb_axes = self.read_scan_nb_axes()
        self.image_nb_per_shot = 2  # assumes 2 images per shot
        self.parameters_labels = self.read_parameters_labels()
        self.total_nb_of_sets_of_params = self.read_total_nb_of_sets_of_params()
        self.param_sets_array, self.data_array = self.read_data()
        self.data_shape_per_param_set = self.data_array.shape[1:]
        
        if print_summary:
            self.summarize_preload()

    def summarize_preload(self) -> None:
        """Print a concise summary of the loaded experiment metadata and arrays."""
        print(f"Experiment folder: {self.experiment_folder_path}")
        print(f"Atom number (ROI centers): {self.atom_number}")
        print(f"Scan axes: {self.scan_nb_axes}")
        print(f"Parameter labels: {self.parameters_labels.tolist()}")
        print(f"Parameter sets loaded: {self.total_nb_of_sets_of_params}")
        print(f"Data shape per parameter set: {self.data_shape_per_param_set}")
        print(f"param_sets_array shape: {self.param_sets_array.shape}")
        print(f"data_array shape: {self.data_array.shape}, dtype: {self.data_array.dtype}, [param_set, shot, image, atom]")
        print(f"Threshold: {self.threshold}")
        for axis in range(self.scan_nb_axes):
            label = self.parameters_labels[axis]
            values = self.get_scan_values(axis)
            print(f"  axis {axis} ({label[0]} {label[1]}): {values.tolist()}")

    def read_atom_number(self) -> int:
        """Return the number of ROI centers from ``rois.json``."""
        rois_path = self.experiment_folder_path / ROIS_NAME
        with open(rois_path, encoding="utf-8") as f:
            rois = json.load(f)
        return len(rois["centers"])

    def read_manifest(self) -> np.ndarray:
        """Parse ``manifest.jsonl`` into an object array of entry dicts.

        Each entry typically contains ``tag``, ``data_file``, ``created_unix``,
        ``data_format_version``, and ``data_layout``.
        """
        manifest_path = self.experiment_folder_path / MANIFEST_NAME
        entries: list[dict] = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return np.array(entries)

    def _resolve_experiment_folder(
        self,
        folder_name: Union[str, os.PathLike],
        *,
        base_dir: Optional[Union[str, os.PathLike]] = None,
    ) -> Path:
        """Accept an absolute path, a relative path, or a folder name next to this file."""
        if base_dir is not None:
            experiment = Path(base_dir) / folder_name
            if not experiment.is_dir():
                raise FileNotFoundError(f"Experiment folder not found: {experiment}")
            return experiment.resolve()

        path = Path(folder_name)
        if path.is_dir():
            return path.resolve()
        here = Path(__file__).resolve().parent
        candidate = here / path
        if candidate.is_dir():
            return candidate.resolve()
        if path.is_absolute():
            raise FileNotFoundError(f"Experiment folder not found: {path}")
        raise FileNotFoundError(
            f"Experiment folder not found: {folder_name!r} "
            f"(tried {path.resolve()} and {candidate})"
        )

    def read_scan_nb_axes(self) -> int:
        """Return how many scan axes appear in a manifest tag (count of ``;``-separated parts)."""
        entry = self.manifest[0]
        tag = entry["tag"]
        if "+" not in tag:
            return 0
        tag_parts = (
            tag.split(PARAM_TAG_SEPARATOR)
            if PARAM_TAG_SEPARATOR in tag
            else [tag]
        )
        return len(tag_parts)

    def read_total_nb_of_sets_of_params(self) -> int:
        """Return the number of manifest lines (= parameter sets / data files)."""
        return len(self.manifest)

    def read_parameters_labels(self) -> np.ndarray:
        """Return ``(device, parameter)`` labels for each scan axis.

        Parsed from the first manifest tag; values (the third ``+``-split field)
        are discarded here — use ``param_sets_array`` for numeric scan values.
        """
        entry = self.manifest[0]
        tag = entry["tag"]
        if "+" not in tag:
            return np.empty((0, 2))
        tag_parts = (
            tag.split(PARAM_TAG_SEPARATOR)
            if PARAM_TAG_SEPARATOR in tag
            else [tag]
        )
        return np.array([param_tag.split("+")[:-1] for param_tag in tag_parts])

    def get_scan_values(self, param_index: int) -> np.ndarray:
        """Return sorted unique values scanned along axis ``param_index``.

        Parameters
        ----------
        param_index
            Column index in ``param_sets_array`` (0-based scan axis).
        """
        return np.unique(self.param_sets_array[:, param_index])

    def read_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all manifest entries into parallel parameter and data arrays.

        Iterates over ``self.manifest``, parses each ``tag`` into
        ``param_sets_array[idx]`` and loads the corresponding ``.npy`` into
        ``data_array[idx]``.

        Returns
        -------
        param_sets_array
            Shape ``(total_nb_of_sets_of_params, scan_nb_axes)``; values are
            strings as stored in the manifest tags.
        data_array
            Float array of shape ``(total_nb_of_sets_of_params, max_shots,
            image_nb_per_shot, atom_number)`` where ``max_shots`` is the largest
            shot count among the files on disk. Entries with fewer shots are
            padded with NaN; entries whose file is missing are entirely NaN.
        """
        param_sets_array = np.empty((self.total_nb_of_sets_of_params, self.scan_nb_axes))
        data_dir = self.experiment_folder_path / DATA_SUBDIR
        blocks: list[Optional[np.ndarray]] = []

        for idx, entry in enumerate(self.manifest):
            if self.scan_nb_axes > 0:
                tag = entry["tag"]
                tag_parts = (
                    tag.split(PARAM_TAG_SEPARATOR)
                    if PARAM_TAG_SEPARATOR in tag
                    else [tag]
                )

                for param_idx, param_tag in enumerate(tag_parts):
                    device, parameter, value = param_tag.split("+")
                    param_sets_array[idx, param_idx] = value

            data_path = data_dir / entry["data_file"]
            blocks.append(np.load(data_path) if data_path.is_file() else None)

        loaded_blocks = [block for block in blocks if block is not None]
        if not loaded_blocks:
            raise FileNotFoundError("No .npy data files found yet for this experiment.")
        max_shots = max(block.shape[0] for block in loaded_blocks)
        data_array = np.full(
            (self.total_nb_of_sets_of_params, max_shots, *loaded_blocks[0].shape[1:]),
            np.nan,
        )
        for idx, block in enumerate(blocks):
            if block is not None:
                data_array[idx, : block.shape[0]] = block
        return param_sets_array, data_array

    @staticmethod
    def get_valid_shots(entry_data: np.ndarray) -> np.ndarray:
        """Return a boolean mask over shots, ``False`` for NaN-padded shots."""
        return ~np.isnan(entry_data).any(axis=(1, 2))

    def resolve_threshold(self, threshold: Optional[int] = None) -> int:
        """Return ``threshold`` if given, otherwise this experiment's ``self.threshold``."""
        return self.threshold if threshold is None else threshold

    def compute_loading_array(self, threshold: Optional[int] = None) -> np.ndarray:
        threshold = self.resolve_threshold(threshold)
        loading_array = np.full(self.total_nb_of_sets_of_params, None, dtype=object)

        for param_idx in range(self.total_nb_of_sets_of_params):
            entry_data = self.data_array[param_idx]
            valid_shots = self.get_valid_shots(entry_data)
            if not valid_shots.any():
                continue

            fluo_1 = entry_data[valid_shots, 0, :]

            loaded = fluo_1 > threshold

            loading_array[param_idx] = loaded.mean()

        return loading_array

    def compute_recapture_average(self, threshold: Optional[int] = None) -> np.ndarray:
        threshold = self.resolve_threshold(threshold)
        recapture_array = np.full(self.total_nb_of_sets_of_params, None, dtype=object)

        # Sites never loaded are NaN in the per-atom array and are left out of the average
        recapture_per_atom = self.compute_recapture_per_atom(threshold)
        has_loaded_site = ~np.isnan(recapture_per_atom).all(axis=1)
        recapture_array[has_loaded_site] = np.nanmean(recapture_per_atom[has_loaded_site], axis=1)

        return recapture_array
    
    def compute_recapture_per_atom(self, threshold: Optional[int] = None) -> np.ndarray:
        threshold = self.resolve_threshold(threshold)
        recapture_per_atom = np.full((self.total_nb_of_sets_of_params, self.atom_number), np.nan)

        for param_idx in range(self.total_nb_of_sets_of_params):
            entry_data = self.data_array[param_idx]
            valid_shots = self.get_valid_shots(entry_data)
            if not valid_shots.any():
                continue
            
            fluo_1 = entry_data[valid_shots, 0, :]
            fluo_2 = entry_data[valid_shots, 1, :]

            loaded  = fluo_1 > threshold
            recap   = (fluo_2 > threshold) & loaded

            with np.errstate(invalid="ignore"):
                recapture_per_atom[param_idx, :] = recap.sum(axis=0) / loaded.sum(axis=0)
        
        return recapture_per_atom

    def auto_plot(self, threshold: Optional[int] = None) -> None:
        recapture = self.compute_recapture_average(threshold)
        mask = np.array([value is not None for value in recapture], dtype=bool)

        x = self.param_sets_array[mask, 0]
        y = np.array(recapture[mask], dtype=float)

        plt.plot(
            x,
            y,
            marker='o',
            linestyle='-',
            label=self.folder_name
        )

        plt.ylabel("Recapture Probability")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc=(1.02, 0.0))
        plt.ylim(-0.03, 1.03)
        plt.xlabel(self.parameters_labels[0])

        return x, y

    def auto_plot_2d(self, threshold: Optional[int] = None) -> None:
        if self.scan_nb_axes != 2:
            raise ValueError(f"Expected 2 scan axes, got {self.scan_nb_axes}")
        recapture = self.compute_recapture_average(threshold)
        x_vals = self.get_scan_values(0)  # amplitude
        y_vals = self.get_scan_values(1)  # time
        grid = np.full((len(y_vals), len(x_vals)), np.nan)
        x_to_i = {v: i for i, v in enumerate(x_vals)}
        y_to_j = {v: j for j, v in enumerate(y_vals)}
        for idx, z in enumerate(recapture):
            if z is None:
                continue
            x, y = self.param_sets_array[idx]
            grid[y_to_j[y], x_to_i[x]] = float(z)
        fig, ax = plt.subplots(figsize=(8, 5))
        mesh = ax.pcolormesh(x_vals, y_vals, grid, shading="auto", vmin=0, vmax=1)
        fig.colorbar(mesh, ax=ax, label="Recapture Probability")
        lab0, lab1 = self.parameters_labels[0], self.parameters_labels[1]
        ax.set_xlabel(f"{lab0[0]} {lab0[1]}")
        ax.set_ylabel(f"{lab1[0]} {lab1[1]}")
        ax.set_title(Path(self.folder_name).name)

        return x_vals, y_vals, grid