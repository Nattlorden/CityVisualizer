"""Synthetic tests for final v0.20 coastal checkpoint."""
import unittest

import cv2
import numpy as np
from shapely.geometry import LineString

from coast_checkpoint import (small_water_proximity, candidate_near_small_water,
                              reject_short_spur)


class CoastCheckpointTests(unittest.TestCase):
    def test_small_pond_isolated_without_changing_water(self):
        water=np.zeros((150,220),np.uint8)
        water[:,130:]=255  # main sea, touches image edge
        cv2.circle(water,(55,70),13,255,-1)  # separate little pond
        original=water.copy()
        small,near,records=small_water_proximity(water,{
            "small_water_min_area_px":20,
            "small_water_max_area_px":2000,"small_water_near_px":12})
        self.assertEqual(int(small[70,55]),255)
        self.assertEqual(int(small[70,180]),0)
        self.assertTrue(np.array_equal(water,original))
        self.assertTrue(candidate_near_small_water(
            LineString([(42,60),(55,45),(68,60)]),near,
            {"small_water_review_max_line_px":130,
             "small_water_near_fraction":.7}))

    def test_ink_supported_quay_not_removed(self):
        water=np.zeros((120,180),np.uint8)
        water[70:,:]=255
        distance=cv2.distanceTransform((water==0).astype(np.uint8),cv2.DIST_L2,3)
        ink=np.zeros_like(water)
        points=[(12,40),(32,40),(32,15),(70,15)]
        cv2.polylines(ink,[np.asarray(points,np.int32)],False,255,1)
        line,changes=reject_short_spur(LineString(points),distance,ink,{})
        self.assertEqual(changes,[])
        self.assertEqual(list(line.coords),points)

    def test_uninked_triangle_point_removed_if_inland(self):
        water=np.zeros((120,130),np.uint8)
        water[90:,:]=255
        distance=cv2.distanceTransform((water==0).astype(np.uint8),cv2.DIST_L2,3)
        points=[(20,67),(25,48),(30,67),(50,67)]
        result,changes=reject_short_spur(
            LineString(points),distance,np.zeros_like(water),
            {"triangle_max_base_px":15,"triangle_min_height_px":7,
             "triangle_min_inland_px":4,"triangle_max_ink_support":.2})
        self.assertEqual(len(changes),1)
        self.assertNotIn((25.0,48.0),list(result.coords))

if __name__=="__main__":
    unittest.main()
