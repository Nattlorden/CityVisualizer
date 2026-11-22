"""Synthetic smoke test for v0.7, no copyrighted test maps required.

Run: .venv\\Scripts\\python.exe MapExtractor\\test_water_refine.py
"""
import unittest

import cv2
import numpy as np

from water_refine import grow_near_water, seed_lakes


class WaterRefinementTests(unittest.TestCase):
    def test_shore_extension_is_local(self):
        image = np.full((90, 90, 3), (60, 120, 105), dtype=np.uint8)
        cv2.rectangle(image, (15, 15), (40, 40), (135, 140, 120), -1)
        cv2.rectangle(image, (70, 70), (80, 80), (135, 140, 120), -1)
        baseline = np.zeros((90, 90), np.uint8)
        cv2.rectangle(baseline, (22, 22), (33, 33), 255, -1)
        permitted = np.full((90, 90), 255, np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sample = hsv[25, 25].astype(int).tolist()
        lo = [max(0, v - d) for v, d in zip(sample, [6, 20, 20])]
        hi = [min(m, v + d) for v, d, m in zip(sample, [6, 20, 20], [179, 255, 255])]
        profile = {"shore_refinement": {"distance_px": 8, "hsv_ranges": [
            {"lower": lo, "upper": hi}]}}
        added = grow_near_water(image, baseline, profile, permitted)
        self.assertGreater(np.count_nonzero(added), 0)
        self.assertEqual(int(added[75, 75]), 0)

    def test_seeded_water_respects_radius(self):
        image = np.full((90, 90, 3), (50, 150, 55), dtype=np.uint8)
        cv2.circle(image, (45, 45), 12, (150, 145, 110), -1)
        profile = {"seeded_water": {"points": [[45, 45]],
                                     "max_radius_px": 18,
                                     "lab_tolerance": [7, 7, 7]}}
        mask, report = seed_lakes(image, profile, np.full((90, 90), 255, np.uint8))
        self.assertGreater(np.count_nonzero(mask), 0)
        self.assertEqual(int(mask[10, 10]), 0)
        self.assertEqual(report[0]["status"], "processed")


if __name__ == "__main__":
    unittest.main()
