"""Synthetic v0.14 land/shore boundary separation tests."""
import unittest

import cv2
import numpy as np

from ink_coast_select import side_score


class TwoSidedCoastTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((190, 240, 3), np.uint8)
        self.image[:, :95] = (65, 150, 85)       # green land, BGR
        self.image[:, 95:158] = (145, 185, 210)  # ochre shore
        self.image[:, 158:] = (190, 140, 80)     # blue water
        cv2.line(self.image, (95, 8), (95, 182), (35, 35, 35), 2)
        cv2.rectangle(self.image, (30, 45), (49, 65), (35, 35, 35), 2)
        self.allowed = np.full((190, 240), 255, np.uint8)
        self.water = np.zeros((190, 240), np.uint8)
        self.water[:, 158:] = 255
        self.ink = cv2.inRange(
            cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), 0, 50)
        self.cfg = {
            "green_hsv_ranges": [{"lower": [19, 35, 43],
                                   "upper": [89, 255, 255]}],
            "shore_hsv_ranges": [{"lower": [8, 12, 65],
                                   "upper": [36, 210, 255]}],
            "color_smoothing_sigma_px": 5,
            "normal_sigma_px": 11,
            "side_offset_px": 11,
            "min_normal_strength": 0.045,
            "min_side_color_fraction": 0.22,
            "min_side_separation": 0.16,
            "max_water_distance_px": 150,
        }

    def test_boundary_selected_but_building_rejected(self):
        selected, _ = side_score(self.image, self.ink, self.water,
                                 self.allowed, self.cfg)
        self.assertGreater(np.count_nonzero(selected[20:170, 94:97]), 50)
        self.assertEqual(np.count_nonzero(selected[42:69, 27:53]), 0)

    def test_no_selected_line_when_shore_evidence_missing(self):
        opts = dict(self.cfg)
        opts["shore_hsv_ranges"] = [{"lower": [0, 250, 250],
                                       "upper": [0, 255, 255]}]
        selected, _ = side_score(self.image, self.ink, self.water,
                                 self.allowed, opts)
        self.assertEqual(int(np.count_nonzero(selected)), 0)


if __name__ == "__main__":
    unittest.main()
