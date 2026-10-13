"""Depth sampling checks that do not require a physical camera or its SDK."""

import json
import unittest

import numpy as np

from depth_estimator import DepthEstimator


class DepthEstimatorTests(unittest.TestCase):
    def test_raw_depth_scale_and_frame_adapter(self):
        raw = np.full((12, 12), 800, dtype=np.uint16)
        class Frame:
            def get_data(self):
                return raw
            def get_units(self):
                return 0.0025
        result = DepthEstimator().estimate_frame(Frame(), [1, 1, 10, 10])
        self.assertEqual(result["status"], "valid")
        self.assertAlmostEqual(result["depth_m"], 2.0)
        self.assertEqual(result["sample_pixel_uv"], [6, 6])
        json.dumps(result, allow_nan=False)

    def test_invalid_readings_and_isolated_outlier(self):
        metric = np.full((7, 7), 2.0)
        metric.flat[:8] = [0, -1, np.nan, np.inf, -np.inf, 0.01, 20, 9]
        original = metric.copy()
        result = DepthEstimator().estimate(metric, [0, 0, 7, 7], depth_scale=1.0)
        self.assertEqual(result["depth_m"], 2.0)
        self.assertEqual(result["depth_mad_m"], 0.0)
        self.assertEqual(result["valid_pixels"], 42)
        np.testing.assert_array_equal(metric, original)

    def test_insufficient_coverage_returns_unavailable(self):
        depth = np.zeros((7, 7), dtype=np.uint16)
        depth.flat[:24] = 2000
        result = DepthEstimator().estimate(depth, [0, 0, 7, 7], depth_scale=0.001)
        self.assertEqual(result["status"], "insufficient_valid_depth")
        self.assertIsNone(result["depth_m"])
        json.dumps(result, allow_nan=False)

    def test_empty_depth(self):
        result = DepthEstimator().estimate(np.zeros((7, 7)), [0, 0, 7, 7], depth_scale=1)
        self.assertEqual(result["valid_pixels"], 0)
        self.assertIsNone(result["depth_m"])

    def test_patch_stays_inside_box_and_image(self):
        depth = np.full((10, 10), 8000, dtype=np.uint16)
        depth[:3, :2] = 1500
        result = DepthEstimator().estimate(depth, [-2, -2, 4, 5], depth_scale=0.001)
        self.assertEqual(result["patch_xywh"], [0, 0, 2, 3])
        self.assertEqual(result["depth_m"], 1.5)
        self.assertEqual(result["total_pixels"], 6)

    def test_outside_box(self):
        depth = np.full((5, 5), 1000, dtype=np.uint16)
        for box in ([5, 0, 2, 2], [-4, 0, 2, 2], [0, 5, 2, 2], [0, -4, 2, 2]):
            with self.subTest(box=box):
                result = DepthEstimator().estimate(depth, box, depth_scale=0.001)
                self.assertEqual(result["status"], "outside_image")
                self.assertIsNone(result["depth_m"])

    def test_single_pixel_and_minimum_samples(self):
        depth = np.full((2, 2), 2000, dtype=np.uint16)
        result = DepthEstimator().estimate(depth, [1, 1, 1, 1], depth_scale=0.001)
        self.assertIsNone(result["depth_m"])
        result = DepthEstimator(patch_radius=0, min_valid_pixels=1).estimate(
            depth, [1, 1, 1, 1], depth_scale=0.001)
        self.assertEqual(result["depth_m"], 2.0)

    def test_invalid_options(self):
        options = [{"patch_radius": -1}, {"patch_radius": 1.5},
                   {"min_valid_pixels": 0}, {"min_depth_m": 0},
                   {"max_depth_m": np.inf}, {"min_depth_m": 2, "max_depth_m": 1},
                   {"min_valid_fraction": 0}, {"min_valid_fraction": np.nan}]
        for option in options:
            with self.subTest(option=option), self.assertRaises(ValueError):
                DepthEstimator(**option)

    def test_invalid_inputs(self):
        estimator = DepthEstimator()
        for image in (np.zeros((2, 2, 3)), np.zeros((0, 2)), np.zeros((2, 2), dtype=complex)):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                estimator.estimate(image, [0, 0, 1, 1], depth_scale=1)
        for box in ([0, 0, 0, 1], [0, 0, 1, -1], [np.nan, 0, 1, 1], [0, 0, 1]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                estimator.estimate(np.ones((2, 2)), box, depth_scale=1)
        for scale in (0, -1, np.inf, np.nan):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                estimator.estimate(np.ones((2, 2)), [0, 0, 1, 1], depth_scale=scale)
        with self.assertRaises(ValueError):
            estimator.estimate_frame(None, [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
