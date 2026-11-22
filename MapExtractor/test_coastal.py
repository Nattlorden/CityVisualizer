"""v0.9 synthetic tests without proprietary reference maps.

Run from repository root:
  .venv\\Scripts\\python.exe -m unittest discover -s MapExtractor -p 'test_coastal.py'
"""
import unittest
import numpy as np
import cv2

from coastal import coastline_segments, shore_growth, build_layers


class CoastalTests(unittest.TestCase):
    def test_map_crop_edge_is_not_shore(self):
        water = np.zeros((100, 120), np.uint8)
        water[20:80, :52] = 255
        allowed = np.full(water.shape, 255, np.uint8)
        segments = coastline_segments(water, allowed,
                                     {"open_boundary_distance_px": 3,
                                      "min_segment_length_px": 2})
        self.assertTrue(segments)
        # The open left edge must not be exported as a shoreline.
        for line, _ in segments:
            for x, y in line.coords:
                self.assertGreater(x, 2)

    def test_growth_never_enters_excluded_region(self):
        image = np.full((80, 100, 3), (125, 135, 100), np.uint8)
        baseline = np.zeros((80, 100), np.uint8)
        baseline[20:50, 20:50] = 255
        allowed = np.full_like(baseline, 255)
        allowed[:, 52:] = 0
        hue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[30, 30]
        low = [max(0, int(hue[i]) - 15) for i in range(3)]
        high = [min([179,255,255][i], int(hue[i]) + 15) for i in range(3)]
        opts = {"max_distance_px": 10, "lab_tolerance": [30, 30, 30],
                "water_hsv_ranges": [{"lower": low, "upper": high}]}
        result, added, _ = shore_growth(image, baseline, allowed, opts)
        self.assertTrue(np.any(added))
        self.assertFalse(np.any(result[:, 52:]))

    def test_empty_baseline_does_not_fill_map(self):
        image = np.full((50, 60, 3), (120, 145, 105), np.uint8)
        baseline = np.zeros((50, 60), np.uint8)
        allowed = np.full_like(baseline, 255)
        opts = {"max_distance_px": 5, "lab_tolerance": [30, 30, 30],
                "water_hsv_ranges": [{"lower": [0, 0, 0],
                                      "upper": [179, 255, 255]}]}
        result, added, _ = shore_growth(image, baseline, allowed, opts)
        self.assertFalse(np.any(result))
        self.assertFalse(np.any(added))


if __name__ == "__main__":
    unittest.main()
