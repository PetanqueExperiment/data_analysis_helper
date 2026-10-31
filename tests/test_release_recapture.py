import unittest

import numpy as np
from scipy.constants import h, k

from data_analysis_helper.release_recapture import (
    fit_release_recapture,
    release_recapture,
)


TRAP = dict(waist_um=0.95, depth_MHz=k * 0.75e-3 / (h * 1e6),
            wavelength_nm=850)


class ReleaseRecaptureTests(unittest.TestCase):
    def test_simulation_normalization_and_reproducibility(self):
        args = dict(**TRAP, samples=3000, seed=12)
        first = release_recapture(50e-6, np.array([0, 20e-6, 50e-6]), **args)
        second = release_recapture(50e-6, np.array([0, 20e-6, 50e-6]), **args)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[0][0], 1)
        self.assertEqual(first[1][0], 0)
        self.assertTrue(np.all((first[0] >= 0) & (first[0] <= 1)))
        self.assertLess(first[0][-1], first[0][1])

    def test_simulation_matches_previous_si_interface(self):
        # Reference probabilities from the original SI-input simulator, with
        # waist=0.95e-6 m, depth=k*0.75e-3 J, and zR=pi*waist**2/850e-9 m.
        references = {
            True: [1.0, 0.6593333333333333, 0.1755, 0.08366666666666667],
            False: [1.0, 0.6595, 0.1755, 0.08366666666666667],
        }
        for zero_point, reference in references.items():
            with self.subTest(zero_point=zero_point):
                probability, error, initial_fraction = release_recapture(
                    50e-6, np.array([0, 10, 30, 50]) * 1e-6, **TRAP,
                    samples=6000, seed=12, zero_point=zero_point,
                )
                np.testing.assert_array_equal(probability, reference)
                np.testing.assert_allclose(error, np.sqrt(probability * (1 - probability) / 6000))
                self.assertEqual(initial_fraction, 1.0)

    def test_recovers_parameters_and_handles_missing_unsorted_data(self):
        times = np.linspace(0, 50, 31)
        # Independent simulation samples and measurement noise for recovery.
        probability, _, _ = release_recapture(50e-6, times * 1e-6, **TRAP,
                                               samples=30000, seed=123)
        data = 0.92 * probability + np.random.default_rng(8).normal(0, 0.003, times.size)
        result = fit_release_recapture(
            np.append(times[::-1], 3)[:, None], np.append(data[::-1], np.nan),
            **TRAP, samples=12000,
        )
        self.assertAlmostEqual(result.temperature_uK, 50, delta=3)
        self.assertAlmostEqual(result.survival, 0.92, delta=0.015)
        np.testing.assert_array_equal(result.time_us, times)
        self.assertTrue(np.all(np.linalg.eigvalsh(result.parameter_covariance) > 0))
        self.assertTrue(np.all(result.curve_ci95[0] <= result.curve_probability))
        self.assertTrue(np.all(result.curve_ci95[1] >= result.curve_probability))
        self.assertTrue(np.all(np.isfinite(result.parameter_ci95)))
        self.assertFalse(result.near_bound)

    def test_rejects_invalid_fit_inputs(self):
        for times, data in [([0, 1], [1, 0.5]),
                            ([[0, 1], [2, 3]], [1, 0.5]),
                            ([0, 1, 2], [1, 0.5]),
                            ([0, 1, 2], [1, 1.1, 0.5]),
                            ([-1, 0, 1], [1, 0.9, 0.5])]:
            with self.subTest(times=times, data=data), self.assertRaises(ValueError):
                fit_release_recapture(times, data, **TRAP)
        with self.assertRaises(ValueError):
            fit_release_recapture([0, 1, 2], [1, 0.8, 0.5], **TRAP, seed=None)

    def test_mass_and_zero_point_can_be_overridden(self):
        times = np.linspace(0, 50, 31)
        mass = 2 * 1.44316e-25
        trap = dict(TRAP, mass=mass)
        probability, _, _ = release_recapture(
            20e-6, times * 1e-6, **trap, samples=12000, seed=42, zero_point=False,
        )
        result = fit_release_recapture(
            times, 0.9 * probability, **TRAP, mass=mass,
            samples=12000, zero_point=False,
        )
        self.assertAlmostEqual(result.temperature_uK, 20, delta=0.1)
        self.assertAlmostEqual(result.survival, 0.9, delta=0.001)

    def test_rejects_invalid_fit_trap_parameters(self):
        for key in ('wavelength_nm', 'depth_MHz', 'waist_um', 'mass'):
            for value in (0, -1, np.nan, np.inf):
                with self.subTest(parameter=key, value=value), self.assertRaises(ValueError):
                    fit_release_recapture(
                        [0, 10, 20], [0.9, 0.5, 0.2], **dict(TRAP, **{key: value}),
                    )

    def test_rejects_invalid_simulation_settings(self):
        with self.assertRaises(ValueError):
            release_recapture(50e-6, [0, 1e-6], **TRAP, samples=0)
        with self.assertRaises(ValueError):
            release_recapture(np.nan, [0, 1e-6], **TRAP)
        for key in ('wavelength_nm', 'depth_MHz', 'waist_um', 'mass'):
            for value in (0, -1, np.nan, np.inf):
                with self.subTest(parameter=key, value=value), self.assertRaises(ValueError):
                    release_recapture(50e-6, [0, 1e-6], **dict(TRAP, **{key: value}))


if __name__ == '__main__':
    unittest.main()
