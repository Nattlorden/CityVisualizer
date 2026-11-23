"""Regression tests for v0.22 tower-anchored black masonry traces."""
import unittest

import cv2
import numpy as np

from wall_trace import (darkness_layers, tower_seeds,
                        thin_masonry, trace_tower_arms)


class WallTraceTests(unittest.TestCase):
    def test_black_wall_excludes_gray_road(self):
        image = np.full((90, 160, 3), 220, np.uint8)
        cv2.line(image, (10, 20), (135, 20), (22, 22, 22), 5)
        cv2.line(image, (10, 65), (135, 65), (110, 110, 110), 6)
        result = darkness_layers(image, {"black_gray_max":75},
                                 np.full(image.shape[:2],255,np.uint8))
        self.assertEqual(int(result["near_black"][20, 50]),255)
        self.assertEqual(int(result["near_black"][65, 50]),0)

    def test_towers_are_anchors_not_forced_connections(self):
        black=np.zeros((110,220),np.uint8)
        towers=np.zeros_like(black)
        cv2.rectangle(towers,(10,25),(25,40),255,-1)
        cv2.rectangle(towers,(150,25),(165,40),255,-1)
        cv2.rectangle(black,(10,25),(25,40),255,-1)
        cv2.rectangle(black,(150,25),(165,40),255,-1)
        # A real wall starts from the first tower but ends halfway.
        cv2.line(black,(25,32),(95,32),255,3)
        anchors,seeds,_=tower_seeds(towers,black,{
            "min_tower_area_px":10,"max_tower_area_px":2000,
            "min_tower_black_fraction":0.4})
        self.assertEqual(len(seeds),2)
        paths=trace_tower_arms(thin_masonry(black),anchors,{
            "anchor_reach_px":7,"max_trace_steps":250,
            "min_wall_trace_length_px":8,
            "branch_degree_stop":5})
        # No route may be fabricated through empty space between the towers.
        for item in paths:
            for x,y in item["line"].coords:
                self.assertLess(x,140)

    def test_non_black_tower_rejected(self):
        towers=np.zeros((60,90),np.uint8)
        towers[20:36,20:36]=255
        black=np.zeros_like(towers)
        _,seeds,rejected=tower_seeds(towers,black,{
            "min_tower_area_px":10,"min_tower_black_fraction":0.6})
        self.assertEqual(seeds,[])
        self.assertTrue(rejected)


if __name__ == "__main__":
    unittest.main()
