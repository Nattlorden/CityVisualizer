"""Synthetic fortification smoke tests. No source map distributed."""
import unittest
import cv2
import numpy as np

from fortifications import analyze, region_mask, structural_ink


class FortificationTests(unittest.TestCase):
    def setUp(self):
        self.image=np.full((130,180,3),(140,175,195),np.uint8)
        # Strong dark masonry with one deliberately interrupted wall.
        cv2.rectangle(self.image,(18,15),(25,110),(25,25,25),-1)
        self.image[58:65,18:26]=(140,175,195)
        # Thick square tower on the wall.
        cv2.rectangle(self.image,(11,30),(35,53),(20,20,20),-1)
        # Large pale structure touching the fortification.
        cv2.rectangle(self.image,(27,75),(72,109),(220,225,225),-1)
        self.profile={"land_boundary_v012":{
            "ink_window":19,"ink_contrast":12,"ink_max":155}}
        self.buildings={"thresholds":{
            "min_red":195,"min_green":185,"min_blue":160,
            "max_red_green":42,"max_green_blue":45},
            "median_kernel":3}
        self.options={"opening_radius_px":1,"absolute_dark_max":70,
                      "wall_min_halfwidth_px":1.0,
                      "tower_min_halfwidth_px":4.0,
                      "restore_wall_radius_px":2,"restore_tower_radius_px":3,
                      "building_wall_proximity_px":8,
                      "large_building_area_px2":150,
                      "min_wall_contact_pixels":3,
                      "gate_max_half_gap_px":4}

    def test_masks_include_wall_tower_and_attached_building(self):
        result=analyze(self.image,self.profile,self.buildings,self.options)
        self.assertGreater(np.count_nonzero(result["wall_candidates"]),30)
        self.assertGreater(np.count_nonzero(result["tower_thickenings"]),20)
        self.assertGreater(np.count_nonzero(result["large_attached_buildings"]),100)

    def test_roi_removes_outside_structure(self):
        roi=region_mask(self.image.shape,{"regions":[
            [[0,0],[90,0],[90,129],[0,129]]]})
        ink=structural_ink(self.image,self.profile["land_boundary_v012"],
                           self.options,roi)
        self.assertEqual(np.count_nonzero(ink[:,91:]),0)

    def test_absence_of_structure_produces_no_towers(self):
        blank=np.full_like(self.image,230)
        result=analyze(blank,self.profile,self.buildings,self.options)
        self.assertEqual(np.count_nonzero(result["tower_thickenings"]),0)


if __name__=="__main__":
    unittest.main()
