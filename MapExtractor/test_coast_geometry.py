"""Regression tests: v0.17 preserves good coast and sharp supported quays."""
import unittest

import cv2
import numpy as np
from shapely.geometry import LineString

from coast_geometry import (audit_polyline, make_ink_neighborhood,
                            round_water_features, near_small_round_feature)


class CoastGeometryTests(unittest.TestCase):
    def test_unsupported_diagonal_is_split(self):
        ink = np.zeros((110, 160), np.uint8)
        cv2.line(ink, (10, 40), (60, 40), 255, 1)
        cv2.line(ink, (90, 65), (140, 65), 255, 1)
        line = LineString([(10, 40), (60, 40), (90, 65), (140, 65)])
        pieces, unsupported, _ = audit_polyline(
            line, make_ink_neighborhood(ink, 1),
            {"max_unsupported_link_px": 14, "min_link_ink_fraction": 0.4})
        self.assertEqual(len(unsupported), 1)
        self.assertEqual(len(pieces), 2)

    def test_sharp_ink_supported_quay_survives(self):
        ink = np.zeros((100, 120), np.uint8)
        points = [(10, 50), (40, 50), (40, 15), (90, 15)]
        cv2.polylines(ink, [np.asarray(points, np.int32)], False, 255, 1)
        line = LineString(points)
        pieces, unsupported, tips = audit_polyline(
            line, make_ink_neighborhood(ink, 1),
            {"max_unsupported_link_px": 12, "min_link_ink_fraction": 0.65})
        self.assertEqual(len(pieces), 1)
        self.assertEqual(len(unsupported), 0)
        self.assertEqual(len(tips), 0)
        self.assertEqual(list(pieces[0].coords), points)

    def test_small_round_water_is_review_not_deletion(self):
        water = np.zeros((140, 180), np.uint8)
        cv2.circle(water, (60, 60), 13, 255, -1)
        initial = water.copy()
        mask, features = round_water_features(water, {
            "round_min_area_px2": 20,
            "round_max_area_px2": 1500,
            "round_min_compactness": 0.3,
        })
        self.assertGreater(len(features), 0)
        self.assertTrue(near_small_round_feature(
            LineString([(46, 58), (50, 45), (68, 45), (74, 58)]),
            mask, {"round_review_radius_px": 12,
                   "round_review_max_line_px": 100,
                   "round_review_fraction": 0.7}))
        self.assertTrue(np.array_equal(water, initial))

    def test_long_natural_coast_is_preserved(self):
        ink = np.zeros((100, 220), np.uint8)
        points = [(10, 65), (40, 63), (85, 58), (130, 60), (190, 55)]
        cv2.polylines(ink, [np.asarray(points, np.int32)], False, 255, 1)
        parts, removed, tips = audit_polyline(
            LineString(points), make_ink_neighborhood(ink, 2),
            {"max_unsupported_link_px": 15, "min_link_ink_fraction": 0.45})
        self.assertEqual(len(parts), 1)
        self.assertFalse(removed)
        self.assertFalse(tips)


if __name__ == "__main__":
    unittest.main()
