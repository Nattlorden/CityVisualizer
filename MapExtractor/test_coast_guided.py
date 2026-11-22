"""Synthetic v0.15 water-guided coast regression tests."""
import unittest

import cv2
import numpy as np

from coast_guided import run, choose_path


class GuidedCoastTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((150, 240, 3), (180, 185, 150), np.uint8)
        self.image[:, :100] = (70, 150, 85)
        self.image[:, 100:160] = (140, 190, 210)
        self.image[:, 160:] = (190, 145, 90)
        cv2.line(self.image, (100, 4), (100, 145), (25, 25, 25), 2)
        cv2.rectangle(self.image, (40, 45), (65, 65), (15, 15, 15), 2)
        self.water = np.zeros((150, 240), np.uint8)
        self.water[:, 160:] = 255
        self.profile = {"land_boundary_v012": {
            "ink_window": 19, "ink_contrast": 13, "ink_max": 155}}
        self.cfg = {"sample_step_px": 3, "min_offset_px": 5,
                    "max_offset_px": 80, "min_supported_samples": 5,
                    "min_contour_points": 30, "map_edge_clearance_px": 2}

    def test_continuity_prefers_real_coast_over_inland_building(self):
        _, lines, _ = run(self.image, self.image, self.profile,
                          self.cfg, self.water)
        xs = [point[0] for obj in lines for point in obj["line"].coords]
        self.assertGreater(len(xs), 0)
        self.assertTrue(all(95 <= x <= 105 for x in xs))

    def test_missing_ink_is_a_gap(self):
        rows = [[(20., (10, i))] for i in range(20)]
        for i in range(6, 11):
            rows[i] = []
        chosen = choose_path(rows, self.cfg)
        self.assertTrue(all(chosen[i] is None for i in range(6, 11)))

if __name__ == "__main__":
    unittest.main()
