"""Synthetic regression tests for v0.12 with separate mainland and island."""
import unittest
import cv2
import numpy as np
from land_boundary_dual import trace_regions, analyze


class DualBoundaryTests(unittest.TestCase):
    def test_two_landmasses_remain_separate(self):
        shape=(180,250)
        green=np.zeros(shape,np.uint8)
        cv2.rectangle(green,(10,15),(105,165),255,-1)
        cv2.circle(green,(180,90),23,255,-1)
        sand=cv2.dilate(green,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
        sand[green>0]=0
        ink=cv2.morphologyEx(green,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
        water=np.full(shape,255,np.uint8)
        water[green>0]=0
        allowed=np.full(shape,255,np.uint8)
        options={"green_smooth_px":0,"min_land_area_px2":50,"sand_near_px":15,
                 "water_near_px":75,"edge_clearance_px":1,"min_line_length_px":10}
        lines,_=trace_regions(green,sand,ink,water,allowed,options)
        xs=[line["geometry"].centroid.x for line in lines]
        self.assertTrue(any(x<120 for x in xs), "Expected mainland boundary")
        self.assertTrue(any(x>145 for x in xs), "Expected island boundary")

    def test_no_false_boundary_without_sand(self):
        green=np.zeros((100,100),np.uint8)
        green[20:80,20:80]=255
        water=np.zeros_like(green)
        water[:,85:]=255
        lines,_=trace_regions(green,np.zeros_like(green),
                              np.zeros_like(green),water,
                              np.full_like(green,255),
                              {"green_smooth_px":0,"min_land_area_px2":10})
        self.assertEqual(len(lines),0)


if __name__=="__main__":
    unittest.main()
