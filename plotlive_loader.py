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
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

DATA_SUBDIR = "data"  # subdirectory holding .npy files
MANIFEST_NAME = "manifest.jsonl"
ROIS_NAME = "rois.json"
PARAM_TAG_SEPARATOR = ";"  # separates scan axes in a manifest tag
DEFAULT_THRESHOLD = 217  # fluorescence counts separating "atom" from "no atom"


class ExperimentData:
    """Load metadata and fluorescence arrays from a single PyCam experiment folder.

    Plotting lives in :mod:`data_analysis_helper.data_plotter`; its functions
    take an ``ExperimentData`` as first argument.

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

    def valid_shots(self) -> np.ndarray:
        """Boolean mask of shape ``(n_param_sets, max_shots)``; ``False`` for NaN-padded shots."""
        return ~np.isnan(self.data_array).any(axis=(2, 3))

    def resolve_threshold(self, threshold: Optional[int] = None) -> int:
        """Return ``threshold`` if given, otherwise this experiment's ``self.threshold``."""
        return self.threshold if threshold is None else threshold

    def _loaded(self, threshold: int) -> np.ndarray:
        """``(n_param_sets, max_shots, atom_number)`` mask: site bright in image 1 of a valid shot."""
        return (self.data_array[:, :, 0, :] > threshold) & self.valid_shots()[:, :, None]

    def compute_loading_array(self, threshold: Optional[int] = None) -> np.ndarray:
        """Loading probability per parameter set, averaged over valid shots and sites.

        Returns a float array of length ``n_param_sets``; NaN where the
        parameter set has no valid shot.
        """
        threshold = self.resolve_threshold(threshold)
        n_samples = self.valid_shots().sum(axis=1) * self.atom_number
        with np.errstate(invalid="ignore"):
            return self._loaded(threshold).sum(axis=(1, 2)) / n_samples

    def compute_recapture_per_atom(self, threshold: Optional[int] = None) -> np.ndarray:
        """Recapture probability per parameter set and site.

        Fraction of shots loaded in image 1 that are still bright in image 2.
        Returns a float array of shape ``(n_param_sets, atom_number)``; NaN
        where the site was never loaded (or the set has no valid shot).
        """
        threshold = self.resolve_threshold(threshold)
        loaded = self._loaded(threshold)
        recaptured = loaded & (self.data_array[:, :, 1, :] > threshold)
        with np.errstate(invalid="ignore"):
            return recaptured.sum(axis=1) / loaded.sum(axis=1)

    def compute_recapture_average(self, threshold: Optional[int] = None) -> np.ndarray:
        """Recapture probability per parameter set, averaged over sites.

        Mean of :meth:`compute_recapture_per_atom` over the sites that were
        loaded at least once. Returns a float array of length
        ``n_param_sets``; NaN where no site was loaded.
        """
        recapture_per_atom = self.compute_recapture_per_atom(threshold)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            return np.nanmean(recapture_per_atom, axis=1)

    def compute_recapture_grid(
        self, threshold: Optional[int] = None, *, x_axis: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Arrange the average recapture of a 2-axis scan on a grid.

        ``x_axis`` (0 or 1) is the scan axis along the grid columns. Returns
        ``(x_vals, y_vals, grid)`` with sorted unique scan values and ``grid``
        of shape ``(len(y_vals), len(x_vals))``; missing points are NaN.
        """
        if self.scan_nb_axes != 2:
            raise ValueError(f"Expected 2 scan axes, got {self.scan_nb_axes}")
        if x_axis not in (0, 1):
            raise ValueError(f"x_axis must be 0 or 1, got {x_axis!r}")
        y_axis = 1 - x_axis
        x_vals = self.get_scan_values(x_axis)
        y_vals = self.get_scan_values(y_axis)
        columns = np.searchsorted(x_vals, self.param_sets_array[:, x_axis])
        rows = np.searchsorted(y_vals, self.param_sets_array[:, y_axis])
        grid = np.full((len(y_vals), len(x_vals)), np.nan)
        grid[rows, columns] = self.compute_recapture_average(threshold)
        return x_vals, y_vals, grid