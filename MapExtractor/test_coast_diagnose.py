"""Synthetic checks for v0.19 non-destructive diagnostics."""
import unittest
import cv2
import numpy as np
from coast_diagnose import component_catalog, ink_support

class DiagnosticTests(unittest.TestCase):
    def test_component_catalog_preserves_separate_small_water(self):
        water = np.zeros((100, 140), np.uint8)
        cv2.circle(water, (25, 30), 9, 255, -1)
        cv2.rectangle(water, (65, 10), (135, 85), 255, -1)
        copy = water.copy()
        _, components = component_catalog(water, {"small_water_area_px": 500})
        self.assertEqual(len(components), 2)
        self.assertEqual(sum(obj["small"] for obj in components), 1)
        self.assertTrue(np.array_equal(water, copy))

    def test_unsupported_shortcut_diagnosed(self):
        ink=np.zeros((100, 120), np.uint8)
        cv2.line(ink, (10, 20), (30, 20), 255, 1)
        self.assertLess(ink_support((30,20), (90,60), ink), 0.1)
        self.assertGreater(ink_support((10,20), (30,20), ink), 0.8)

if __name__ == "__main__":
    unittest.main()
