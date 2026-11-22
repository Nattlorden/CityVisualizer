"""v0.11 synthetic tests; no copyrighted source imagery needed."""
import unittest
import cv2
import numpy as np
from land_boundary import extract_ink, candidate_lines

class LandBoundaryTests(unittest.TestCase):
    def test_detects_dark_stroke_near_land_sand(self):
        img = np.zeros((140,160,3), np.uint8)
        img[:] = (160,195,205)  # sandy light background
        img[:, :76] = (80,155,95)  # greenish land
        cv2.line(img,(76,20),(76,120),(30,35,35),2)
        water=np.zeros((140,160),np.uint8)
        water[:,100:]=255
        allowed=np.full_like(water,255)
        params={"green_land_hsv_ranges":[{"lower":[25,20,25],"upper":[90,255,255]}],
                "shore_sand_hsv_ranges":[{"lower":[6,5,40],"upper":[40,240,255]}],
                "search_width_px":70,"min_land_region_px":200,
                "land_proximity_px":10,"sand_proximity_px":12}
        masks=extract_ink(img,water,allowed,params)
        self.assertGreater(int(np.count_nonzero(
            masks["inner_boundary_candidates"][20:120,72:81])),10)

    def test_no_candidates_without_dark_ink(self):
        image=np.full((60,80,3),220,np.uint8)
        water=np.zeros((60,80),np.uint8)
        water[:,55:]=255
        allowed=np.full_like(water,255)
        opts={"green_land_hsv_ranges":[{"lower":[0,0,0],"upper":[179,255,255]}],
              "shore_sand_hsv_ranges":[{"lower":[0,0,0],"upper":[179,255,255]}]}
        masks=extract_ink(image,water,allowed,opts)
        self.assertEqual(int(np.count_nonzero(masks["inner_boundary_candidates"])),0)
        self.assertEqual(candidate_lines(masks["inner_boundary_candidates"],opts),[])

if __name__=="__main__":
    unittest.main()
