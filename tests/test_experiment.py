import json
import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data_analysis_helper import (  # noqa: E402
    ExperimentData,
    plot_fluo_recapture,
    plot_fluorescence,
    plot_loading,
    plot_recapture,
    plot_recapture_2d,
    plot_recapture_per_atom,
)
from data_analysis_helper.experiment import _parse_tag  # noqa: E402

THRESHOLD = 100
BRIGHT, DARK = 200.0, 0.0


def shots(loaded, recaptured):
    """Build a ``(shots, 2, atoms)`` array from 0/1 masks of image 1 and image 2."""
    loaded, recaptured = np.asarray(loaded), np.asarray(recaptured)
    return np.stack([np.where(loaded, BRIGHT, DARK), np.where(recaptured, BRIGHT, DARK)], axis=1)


def write_experiment(folder, entries, atoms):
    """Write an experiment folder; ``entries`` is a list of ``(tag, array or None)``."""
    (folder / "data").mkdir(parents=True)
    (folder / "rois.json").write_text(json.dumps({"centers": [[0, 0]] * atoms}))
    lines = []
    for idx, (tag, data) in enumerate(entries):
        lines.append(json.dumps({"tag": tag, "data_file": f"{idx}.npy"}))
        if data is not None:
            np.save(folder / "data" / f"{idx}.npy", data)
    (folder / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    return folder


class ExperimentTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        plt.close("all")
        self._tmp.cleanup()

    def load(self, name, entries, atoms):
        write_experiment(self.root / name, entries, atoms)
        return ExperimentData(self.root / name, threshold=THRESHOLD, print_summary=False)


class OneAxisScanTests(ExperimentTestCase):
    """3 sites; one entry per edge case."""

    def setUp(self):
        super().setUp()
        self.exp = self.load("scan1d", [
            # Site 2 never loaded; an image-2 count on it must not count as recapture.
            ("AOM+amplitude+0.1", shots(
                loaded=[[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0]],
                recaptured=[[1, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]],
            )),
            ("AOM+amplitude+0.2", None),  # file missing
            # Only 2 shots: padded with NaN to 4, padding must not dilute loading.
            ("AOM+amplitude+0.3", shots(loaded=np.ones((2, 3)), recaptured=np.ones((2, 3)))),
            ("AOM+amplitude+0.4", shots(loaded=np.zeros((4, 3)), recaptured=np.zeros((4, 3)))),
        ], atoms=3)

    def test_metadata(self):
        exp = self.exp
        self.assertEqual(exp.atom_number, 3)
        self.assertEqual(exp.scan_nb_axes, 1)
        self.assertEqual(exp.total_nb_of_sets_of_params, 4)
        self.assertEqual(exp.image_nb_per_shot, 2)
        self.assertEqual(exp.data_shape_per_param_set, (4, 2, 3))
        self.assertEqual(exp.parameters_labels.tolist(), [["AOM", "amplitude"]])
        self.assertEqual(exp.param_sets_array.dtype, float)
        np.testing.assert_array_equal(exp.param_sets_array[:, 0], [0.1, 0.2, 0.3, 0.4])

    def test_missing_file_and_padding_are_nan(self):
        self.assertTrue(np.isnan(self.exp.data_array[1]).all())
        self.assertTrue(np.isnan(self.exp.data_array[2, 2:]).all())
        np.testing.assert_array_equal(
            self.exp.valid_shots(),
            [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]],
        )

    def test_loading(self):
        np.testing.assert_array_equal(self.exp.compute_loading_array(), [0.5, np.nan, 1.0, 0.0])

    def test_recapture_per_atom(self):
        np.testing.assert_array_equal(self.exp.compute_recapture_per_atom(), [
            [0.75, 0.5, np.nan],
            [np.nan, np.nan, np.nan],
            [1.0, 1.0, 1.0],
            [np.nan, np.nan, np.nan],
        ])

    def test_recapture_average_is_float_with_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            average = self.exp.compute_recapture_average()
        self.assertEqual(average.dtype, float)
        np.testing.assert_array_equal(average, [0.625, np.nan, 1.0, np.nan])

    def test_threshold_argument_overrides_default(self):
        # Above every count: nothing is loaded anywhere.
        np.testing.assert_array_equal(self.exp.compute_loading_array(threshold=500),
                                      [0.0, np.nan, 0.0, 0.0])

    def test_grid_needs_two_axes(self):
        with self.assertRaises(ValueError):
            self.exp.compute_recapture_grid()


class TwoAxisScanTests(ExperimentTestCase):
    def setUp(self):
        super().setUp()

        def recaptured_in(n):  # 1 site, 4 shots, all loaded, n recaptured
            return shots(loaded=np.ones((4, 1)), recaptured=(np.arange(4) < n)[:, None])

        # Unsorted on purpose; (x=2, y=20) is never measured.
        self.exp = self.load("scan2d", [
            ("A+x+2;B+y+10", recaptured_in(2)),
            ("A+x+1;B+y+10", recaptured_in(1)),
            ("A+x+1;B+y+20", recaptured_in(3)),
        ], atoms=1)

    def test_labels_and_values(self):
        self.assertEqual(self.exp.parameters_labels.tolist(), [["A", "x"], ["B", "y"]])
        np.testing.assert_array_equal(self.exp.param_sets_array, [[2, 10], [1, 10], [1, 20]])
        np.testing.assert_array_equal(self.exp.get_scan_values(1), [10, 20])

    def test_grid(self):
        x, y, grid = self.exp.compute_recapture_grid()
        np.testing.assert_array_equal(x, [1, 2])
        np.testing.assert_array_equal(y, [10, 20])
        np.testing.assert_array_equal(grid, [[0.25, 0.5], [0.75, np.nan]])

    def test_grid_with_swapped_axes(self):
        x, y, grid = self.exp.compute_recapture_grid(x_axis=1)
        np.testing.assert_array_equal(x, [10, 20])
        np.testing.assert_array_equal(grid, [[0.25, 0.75], [0.5, np.nan]])
        with self.assertRaises(ValueError):
            self.exp.compute_recapture_grid(x_axis=2)


class SubsetTests(ExperimentTestCase):
    """3-axis scan: A x in (1, 2), B y in (10, 20), C z in (0.5, 1, 2); one file missing."""

    def setUp(self):
        super().setUp()
        entries = []
        for i, (x, y, z) in enumerate((x, y, z) for z in (2, 0.5, 1) for x in (1, 2) for y in (10, 20)):
            # 1 site, 8 shots, all loaded, a different number recaptured per entry.
            data = shots(loaded=np.ones((8, 1)), recaptured=(np.arange(8) < i % 9)[:, None])
            entries.append((f"A+x+{x};B+y+{y};C+z+{z}", None if i == 5 else data))
        self.exp = self.load("scan3d", entries, atoms=1)

    def test_split_by_axis(self):
        parent_average = self.exp.compute_recapture_average()
        parts = self.exp.split_by_axis(2)
        self.assertEqual(list(parts), [0.5, 1.0, 2.0])
        for z, sub in parts.items():
            with self.subTest(z=z):
                rows = self.exp.param_sets_array[:, 2] == z
                self.assertEqual(sub.scan_nb_axes, 2)
                self.assertEqual(sub.total_nb_of_sets_of_params, 4)
                self.assertEqual(sub.parameters_labels.tolist(), [["A", "x"], ["B", "y"]])
                np.testing.assert_array_equal(sub.param_sets_array, self.exp.param_sets_array[rows, :2])
                np.testing.assert_array_equal(sub.data_array, self.exp.data_array[rows])
                np.testing.assert_array_equal(sub.compute_recapture_average(), parent_average[rows])
                self.assertEqual(sub.fixed_parameters, [("C", "z", z)])
                self.assertEqual(sub.label, f"{self.exp.experiment_folder_path} [C z = {z:g}]")
                _, _, grid = sub.compute_recapture_grid()
                self.assertEqual(grid.shape, (2, 2))
        # The missing file (entry 5: z=0.5, x=1, y=20) shows up as NaN in its slice only.
        self.assertTrue(np.isnan(parts[0.5].compute_recapture_grid()[2][1, 0]))

    def test_original_is_unchanged(self):
        before = (self.exp.param_sets_array.copy(), self.exp.parameters_labels.copy())
        self.exp.split_by_axis(0)
        self.exp.fix_axis(2, 1).fix_axis(0, 2)
        self.assertEqual(self.exp.scan_nb_axes, 3)
        self.assertEqual(self.exp.total_nb_of_sets_of_params, 12)
        self.assertEqual(self.exp.fixed_parameters, [])
        self.assertEqual(self.exp.label, str(self.exp.experiment_folder_path))
        np.testing.assert_array_equal(self.exp.param_sets_array, before[0])
        np.testing.assert_array_equal(self.exp.parameters_labels, before[1])

    def test_chained_fix_axis_gives_1d_scan(self):
        line = self.exp.fix_axis(2, 1).fix_axis(0, 2)
        self.assertEqual(line.scan_nb_axes, 1)
        self.assertEqual(line.parameters_labels.tolist(), [["B", "y"]])
        np.testing.assert_array_equal(line.param_sets_array[:, 0], [10, 20])
        self.assertEqual(line.fixed_parameters, [("C", "z", 1.0), ("A", "x", 2.0)])
        self.assertTrue(line.label.endswith("[C z = 1, A x = 2]"))
        ax = plot_recapture(line)
        self.assertEqual(ax.get_legend().get_texts()[0].get_text(), line.label)

    def test_fix_axis_rejects_bad_axis_or_value(self):
        with self.assertRaisesRegex(ValueError, r"scanned values: \[0.5, 1.0, 2.0\]"):
            self.exp.fix_axis(2, 3)
        for axis in (-1, 3):
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                self.exp.fix_axis(axis, 1)

    def test_subset_keeps_axes(self):
        sub = self.exp.subset(self.exp.param_sets_array[:, 0] == 1)
        self.assertEqual(sub.scan_nb_axes, 3)
        self.assertEqual(sub.total_nb_of_sets_of_params, 6)
        self.assertEqual(sub.fixed_parameters, [])
        np.testing.assert_array_equal(self.exp.subset([3, 0]).param_sets_array,
                                      self.exp.param_sets_array[[3, 0]])
        self.assertEqual(self.exp.subset(slice(0, 2)).total_nb_of_sets_of_params, 2)
        with self.assertRaises(ValueError):
            self.exp.subset(np.zeros(12, dtype=bool))


class TagParsingTests(ExperimentTestCase):
    def test_parse_tag(self):
        self.assertEqual(_parse_tag("AOM+amplitude+0.43"), [("AOM", "amplitude", "0.43")])
        self.assertEqual(_parse_tag("A+x+1e+03;B+y+-2"), [("A", "x", "1e+03"), ("B", "y", "-2")])
        self.assertEqual(_parse_tag("no_scan"), [])
        with self.assertRaises(ValueError):
            _parse_tag("A+x")

    def test_no_scan(self):
        exp = self.load("noscan", [("running", shots([[1]], [[1]]))], atoms=1)
        self.assertEqual(exp.scan_nb_axes, 0)
        self.assertEqual(exp.parameters_labels.shape, (0, 2))
        self.assertEqual(exp.param_sets_array.shape, (1, 0))

    def test_mismatched_axes_raise(self):
        data = shots([[1]], [[1]])
        for name, second_tag in [("reordered", "B+y+2;A+x+1"), ("missing", "A+x+1"),
                                 ("renamed", "A+x+1;C+y+2")]:
            with self.subTest(name), self.assertRaises(ValueError):
                self.load(name, [("A+x+1;B+y+2", data), (second_tag, data)], atoms=1)

    def test_empty_manifest_and_no_data_raise(self):
        with self.assertRaises(ValueError):
            self.load("empty", [], atoms=1)
        with self.assertRaises(FileNotFoundError):
            self.load("nodata", [("A+x+1", None)], atoms=1)


class FolderResolutionTests(ExperimentTestCase):
    def test_base_dir_and_missing_folder(self):
        write_experiment(self.root / "exp", [("A+x+1", shots([[1]], [[1]]))], atoms=1)
        exp = ExperimentData("exp", base_dir=self.root, print_summary=False)
        self.assertEqual(exp.experiment_folder_path, (self.root / "exp").resolve())
        with self.assertRaises(FileNotFoundError):
            ExperimentData("missing", base_dir=self.root, print_summary=False)
        with self.assertRaises(FileNotFoundError):
            ExperimentData(self.root / "missing", print_summary=False)


class PlotSmokeTests(ExperimentTestCase):
    """Every plot runs, returns its axes/figure, and forwards style kwargs."""

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(0)
        self.exp1d = self.load("p1d", [
            (f"AOM+amplitude+{a}", rng.normal(THRESHOLD + 30, 60, (10, 2, 4))) for a in (0.3, 0.1, 0.2)
        ], atoms=4)
        self.exp2d = self.load("p2d", [
            (f"A+x+{x};B+y+{y}", rng.normal(THRESHOLD + 30, 60, (10, 2, 4)))
            for x in (1, 2, 3) for y in (10, 20)
        ], atoms=4)

    def test_curves(self):
        ax = plot_recapture(self.exp1d, marker="s", color="C3")
        self.assertEqual(ax.lines[0].get_marker(), "s")
        self.assertEqual(ax.get_xlabel(), "AOM amplitude")
        self.assertEqual(ax.get_legend().get_texts()[0].get_text(),
                         str(self.exp1d.experiment_folder_path))
        self.assertIs(plot_loading(self.exp1d, ax=ax, label="loading"), ax)
        self.assertEqual(len(ax.lines), 2)  # overlays on the same axes

    def test_unsorted_scan_is_plotted_in_x_order(self):
        # Manifest order 0.3, 0.1, 0.2, 0.1 (repeat); each set has a distinct recapture.
        def recaptured_in(n):
            return shots(loaded=np.ones((4, 1)), recaptured=(np.arange(4) < n)[:, None])

        exp = self.load("unsorted", [
            ("AOM+amplitude+0.3", recaptured_in(3)),
            ("AOM+amplitude+0.1", recaptured_in(1)),
            ("AOM+amplitude+0.2", recaptured_in(2)),
            ("AOM+amplitude+0.1", recaptured_in(4)),
        ], atoms=1)
        line = plot_recapture(exp).lines[-1]
        np.testing.assert_array_equal(line.get_xdata(), [0.1, 0.1, 0.2, 0.3])
        np.testing.assert_array_equal(line.get_ydata(), [0.25, 1.0, 0.5, 0.75])  # repeats keep order

        # Series skip sets with no counts in their category (here "Background" is empty).
        for fig in (plot_fluorescence(exp, 1), plot_fluo_recapture(exp)):
            xs = [np.asarray(c.lines[0].get_xdata()) for ax in fig.axes for c in ax.containers]
            self.assertGreater(sum(len(x) for x in xs), 0)
            for x in xs:
                self.assertTrue(np.all(np.diff(x) >= 0), x)

    def test_per_atom_and_2d(self):
        _, ax = plt.subplots()
        self.assertIs(plot_recapture_per_atom(self.exp1d, ax=ax), ax)
        plot_recapture_per_atom(self.exp1d, style="curve", ax=plt.figure().gca())
        self.assertEqual(len(plot_recapture_2d(self.exp2d, style="curve").lines), 2)
        plot_recapture_2d(self.exp2d, vmin=0, vmax=1)
        with self.assertRaises(ValueError):
            plot_recapture_2d(self.exp2d, style="bogus")
        with self.assertRaises(ValueError):
            plot_recapture_per_atom(self.exp2d)

    def test_fluorescence_figures(self):
        fig = plt.figure()
        self.assertIs(plot_fluorescence(self.exp1d, 0, fig=fig, capsize=2), fig)
        self.assertEqual(len(fig.axes), 3)
        plot_fluo_recapture(self.exp1d)
        with self.assertRaises(TypeError):
            plot_fluo_recapture(self.exp1d, threshold_loading=100)


if __name__ == "__main__":
    unittest.main()
