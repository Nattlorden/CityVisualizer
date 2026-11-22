"""Offline smoke tests for v0.10, using synthetic images only."""
import unittest
import cv2
import numpy as np
from ink_coast import ink_lines, infer_water, extract_shores

class InkCoastTests(unittest.TestCase):
    def test_ink_recognizes_dark_outline(self):
        img=np.full((90,120,3), (180,200,210), np.uint8)
        cv2.rectangle(img,(20,20),(75,65),(35,35,35),2)
        ink=ink_lines(img,{"local_window_px":15,"min_dark_contrast":15,
                            "max_ink_brightness":140})
        self.assertGreater(np.count_nonzero(ink[19:24,19:76]),30)
        self.assertEqual(int(ink[40,40]),0)

    def test_open_map_edge_not_shoreline(self):
        water=np.zeros((90,110),np.uint8)
        water[15:75,:50]=255
        allowed=np.full_like(water,255)
        ink=np.zeros_like(water)
        features=extract_shores(water,allowed,ink,{"map_edge_clearance_px":3})
        self.assertGreater(len(features),0)
        for feature in features:
            for x,y in feature["geometry"]["coordinates"]:
                self.assertGreater(x,2)

    def test_watershed_respects_roi(self):
        img=np.full((80,100,3),(120,145,95),np.uint8)
        base=np.zeros((80,100),np.uint8)
        base[25:55,20:48]=255
        allowed=np.full_like(base,255)
        allowed[:,75:]=0
        cfg={"water_hsv_ranges":[{"lower":[0,0,0],"upper":[179,255,255]}],
             "land_hsv_ranges":[],"search_distance_px":12}
        result,blue=infer_water(img,base,allowed,np.zeros_like(base),cfg)
        self.assertEqual(int(np.count_nonzero(result[:,75:])),0)
        self.assertGreater(np.count_nonzero(result),0)

if __name__=="__main__":
    unittest.main()
