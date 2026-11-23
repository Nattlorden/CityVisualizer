"""Synthetic v0.18 matching regression tests.

No copyrighted Hârn imagery is needed to check transitions and closed water.
"""
import unittest

import cv2
import numpy as np

from coast_matching import (choose_spatial_path, small_closed_contour,
                            transition_support)


class MatchingGeometryTests(unittest.TestCase):
    def setUp(self):
        self.base={"gap_penalty":1.65,"ink_reward":2.6,
                   "distance_penalty":0.025,"distance_jump_penalty":0.13,
                   "max_step_offset_change_px":50}
        self.options={"ink_support_radius_px":1,"max_point_step_px":35,
                      "verify_step_from_px":10,
                      "min_ink_link_fraction":0.45}

    def test_unsupported_diagonal_becomes_gap(self):
        ink=np.zeros((100,180),np.uint8)
        cv2.line(ink,(10,20),(40,20),255,1)
        cv2.line(ink,(60,45),(90,45),255,1)
        rows=[[(20.,(10,20))],[(20.,(40,20))],
              [(20.,(60,45))],[(20.,(90,45))]]
        chosen,stats=choose_spatial_path(rows,ink,self.base,self.options)
        self.assertGreater(stats["rejected_unsupported"],0)
        self.assertTrue(chosen[1] is None or chosen[2] is None)

    def test_supported_sharp_quay_remains_connected(self):
        ink=np.zeros((110,160),np.uint8)
        cv2.polylines(ink,[np.asarray(
            [(10,70),(45,70),(45,35),(85,35)],np.int32)],False,255,1)
        rows=[[(20.,(10,70))],[(20.,(45,70))],
              [(20.,(45,35))],[(20.,(85,35))]]
        chosen,stats=choose_spatial_path(rows,ink,self.base,
            {**self.options,"max_point_step_px":45})
        self.assertTrue(all(row is not None for row in chosen))
        self.assertEqual(stats["rejected_unsupported"],0)

    def test_small_exterior_closed_water_flagged(self):
        points=np.asarray([(20,10),(30,12),(40,20),(38,30),
                           (30,40),(20,38),(10,30),(12,20)],float)
        contour={"points":points,"water_hole":False,"closed":True}
        self.assertTrue(small_closed_contour(contour,{}))
        self.assertFalse(small_closed_contour(
            {**contour,"water_hole":True},{}))
        self.assertFalse(small_closed_contour(
            {**contour,"closed":False},{}))

    def test_unsupported_empty_ink_is_zero(self):
        self.assertEqual(transition_support(
            (0,0),(15,15),np.zeros((40,40),bool)),0.0)


if __name__=="__main__":
    unittest.main()
