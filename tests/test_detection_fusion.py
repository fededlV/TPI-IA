"""Model-free tests for T4 (S4): per-detector/per-view monotonic calibration, anti-collision WBF and the
EXP-02C blind fusion path."""

import json
import unittest

import numpy as np

from src import detection_fusion as fu
from src.perception_detectors import Detection


class TestQuantileCalibrator(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.scores = rng.uniform(0.05, 0.40, size=500)
        self.cal = fu.QuantileCalibrator.fit(self.scores, tau=0.175, floor=0.05)

    def test_tau_maps_exactly_to_the_canonical_score(self):
        self.assertEqual(self.cal(0.175), fu.CANONICAL_SCORE)
        self.assertEqual(fu.CANONICAL_SCORE, 0.5)

    def test_floor_and_max_map_to_the_ends_and_outside_values_are_clipped(self):
        self.assertEqual(self.cal(0.05), 0.0)
        self.assertEqual(self.cal(float(self.scores.max())), 1.0)
        self.assertEqual(self.cal(0.01), 0.0)
        self.assertEqual(self.cal(0.99), 1.0)

    def test_map_is_strictly_monotone_inside_the_fitted_range(self):
        grid = np.linspace(0.05, float(self.scores.max()), 400)
        self.assertTrue(np.all(np.diff(self.cal(grid)) > 0))

    def test_operating_point_is_preserved(self):
        """Monotone + tau -> 0.5: thresholding calibrated scores at 0.5 keeps exactly the raw scores >= tau."""
        self.assertTrue(np.array_equal(self.cal(self.scores) >= 0.5, self.scores >= 0.175))

    def test_calibrated_scores_are_roughly_uniform_on_each_side(self):
        """Quantile map: half of the [0, 0.5] range per half of the below-tau mass."""
        below = self.scores[self.scores < 0.175]
        self.assertAlmostEqual(float(np.mean(self.cal(below) < 0.25)), 0.5, delta=0.05)

    def test_json_round_trip(self):
        back = fu.QuantileCalibrator.from_dict(json.loads(json.dumps(self.cal.to_dict())))
        self.assertEqual(back, self.cal)
        np.testing.assert_array_equal(back(self.scores), self.cal(self.scores))

    def test_ties_still_give_a_strictly_increasing_map(self):
        cal = fu.QuantileCalibrator.fit([0.1] * 50 + [0.3] * 50 + [0.6], tau=0.3, floor=0.1)
        self.assertTrue(np.all(np.diff(cal.knots_x) > 0) and np.all(np.diff(cal.knots_y) > 0))
        self.assertEqual(cal(0.3), 0.5)

    def test_degenerate_inputs_fail_fast(self):
        with self.assertRaises(ValueError):
            fu.QuantileCalibrator.fit([], tau=0.2, floor=0.05)
        with self.assertRaises(ValueError):
            fu.QuantileCalibrator.fit([0.1, 0.2], tau=0.04, floor=0.05)  # tau below the floor
        with self.assertRaises(ValueError):
            fu.QuantileCalibrator.fit([0.1, 0.2], tau=0.2, floor=0.05)   # tau must be below the max score


class TestCalibrateDetections(unittest.TestCase):
    def test_each_detector_and_view_kind_uses_its_own_map_and_keeps_the_raw_score(self):
        cals = {"owlv2:global": fu.QuantileCalibrator((0.05, 0.2, 0.6), (0.0, 0.5, 1.0)),
                "owlv2:tile": fu.QuantileCalibrator((0.05, 0.3, 0.6), (0.0, 0.5, 1.0)),
                "mmgdino:global": fu.QuantileCalibrator((0.1, 0.4, 0.9), (0.0, 0.5, 1.0))}
        dets = [Detection(1, 0.2, (0, 0, 10, 10), "owlv2:global"), Detection(1, 0.3, (0, 0, 10, 10), "owlv2:tile2"),
                Detection(1, 0.4, (0, 0, 10, 10), "mmgdino:global")]
        out = fu.calibrate(dets, cals)
        self.assertEqual([d.score for d in out], [0.5, 0.5, 0.5])
        self.assertEqual([d.raw_score for d in out], [0.2, 0.3, 0.4])
        self.assertEqual([d.source for d in out], [d.source for d in dets])

    def test_missing_calibrator_fails_fast(self):
        with self.assertRaises(KeyError):
            fu.calibrate([Detection(1, 0.2, (0, 0, 10, 10), "mmgdino:tile0")], {})

    def test_calibration_key_and_retagging(self):
        self.assertEqual(fu.calibration_key("owlv2:tile3"), "owlv2:tile")
        self.assertEqual(fu.calibration_key("owlv2l:global"), "owlv2l:global")
        self.assertEqual(fu.detector_of("mmgdino:tile1"), "mmgdino")
        out = fu.retag([Detection(1, 0.2, (0, 0, 1, 1), "owlv2:tile1")], "owlv2l")
        self.assertEqual(out[0].source, "owlv2l:tile1")


if __name__ == "__main__":
    unittest.main()
