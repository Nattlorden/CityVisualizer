"""v0.23 synthetic regression tests; no copyrighted map imagery."""
import unittest

import cv2
import numpy as np

from wall_trace_building_aware import (
    building_roofs_v05, build_suppression, variants, trace_variants)


class BuildingAwareWallTests(unittest.TestCase):
    def test_v05_recovers_accepted_roof_not_flecks(self):
        image = np.full((110, 150, 3), (70, 155, 85), np.uint8)
        cv2.rectangle(image, (40, 35), (65, 60), (220, 222, 225), -1)
        image[80, 115] = (220, 222, 225)
        config = {
            "v04_color_normalization": {"strength": 0},
            "thresholds": {"min_red": 195, "min_green": 185,
                           "min_blue": 160, "max_red_green": 42,
                           "max_green_blue": 45},
            "v02_filters": {"min_area_px": 8, "max_area_px": 14000,
                            "min_fill_ratio": .2,
                            "min_bbox_side_px": 2},
            "median_kernel": 1, "closing_kernel": 0,
        }
        mask, count = building_roofs_v05(image, config)
        self.assertEqual(count, 1)
        self.assertEqual(int(mask[45, 50]), 255)
        self.assertEqual(int(mask[80, 115]), 0)

    def test_roof_outline_suppressed_but_tower_preserved(self):
        roofs = np.zeros((100, 140), np.uint8)
        roofs[25:65, 35:85] = 255
        towers = np.zeros_like(roofs)
        towers[40:53, 72:89] = 255
        buildings = np.zeros_like(roofs)
        buildings[75:92, 97:122] = 255
        cfg = {"building_outline_pad_px": 3,
               "preserve_tower_radius_px": 4,
               "preserve_attached_buildings": True,
               "attached_building_pad_px": 2}
        masks = build_suppression(roofs, towers, buildings, cfg)
        self.assertEqual(int(masks["suppressed_building_outlines"][25, 35]), 255)
        self.assertEqual(int(masks["suppressed_building_outlines"][45, 78]), 0)
        self.assertEqual(int(masks["protected_attached_buildings"][80, 105]), 255)

    def test_hard_vs_soft_and_same_baseline(self):
        black = np.full((70, 110), 255, np.uint8)
        gray = np.full_like(black, 65)
        gray[:, :10] = 30  # some unrelated, very dark black
        gray[:, 40:50] = 25  # a house-adjacent truly black wall
        suppress = np.zeros_like(black)
        suppress[:, 20:70] = 255
        modes = variants(black, gray,
                         {"suppressed_building_outlines": suppress},
                         {"soft_black_gray_max": 43})
        self.assertTrue(np.array_equal(modes["baseline_v022"], black))
        self.assertEqual(int(modes["hard_mask"][30, 45]), 0)
        self.assertEqual(int(modes["soft_darkness"][30, 45]), 255)
        self.assertEqual(int(modes["soft_darkness"][30, 25]), 0)
        self.assertEqual(int(modes["hard_mask"][30, 5]), 255)

    def test_no_new_bridge_to_other_tower(self):
        black = np.zeros((100, 190), np.uint8)
        towers = np.zeros_like(black)
        towers[40:54, 12:26] = 255
        towers[40:54, 140:154] = 255
        black[towers > 0] = 255
        cv2.line(black, (25, 46), (75, 46), 255, 2)
        result = trace_variants(
            {"baseline_v022": black, "hard_mask": black.copy(),
             "soft_darkness": black.copy()},
            towers,
            {"anchor_reach_px": 5, "min_wall_trace_length_px": 8})
        for part in result.values():
            for route in part["routes"]:
                for x, y in route["line"].coords:
                    self.assertTrue(x < 110 or x > 130)


if __name__ == "__main__":
    unittest.main()
