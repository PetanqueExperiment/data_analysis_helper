import unittest

import numpy as np

from data_analysis_helper.arena import scan_grating_um


class FakeExperiment:
    """Stands in for ``ExperimentData``: only ``read_static_settings`` is used."""

    def __init__(self, settings):
        self.settings = settings

    def read_static_settings(self):
        if self.settings is None:
            raise FileNotFoundError("no sequence.json")
        return self.settings


class ScanGratingTests(unittest.TestCase):
    def test_grating_from_slm_settings(self):
        exp = FakeExperiment({"SLM_Controller": {"grating_x": 41.52, "grating_y": 51.58, "defocus": 2.22}})
        np.testing.assert_array_equal(scan_grating_um(exp), [41.52, 51.58])

    def test_none_when_not_recorded(self):
        self.assertIsNone(scan_grating_um(FakeExperiment(None)))           # no sequence.json
        self.assertIsNone(scan_grating_um(FakeExperiment({"hero_HWP": {"angle": 1.0}})))  # no SLM row


if __name__ == "__main__":
    unittest.main()
