"""Synthetic checks for v0.16 cleanup without changing v0.7."""
import unittest
import numpy as np
import cv2

from coast_refine import remove_spikes, segment_evidence, curvature_score
from shapely.geometry import LineString


class RefineTests(unittest.TestCase):
    def test_short_backtracking_spike_is_removed(self):
        points=[(0,10),(4,10),(4,4),(4,10),(8,10),(12,10)]
        cleaned, n = remove_spikes(points, {
            "spike_span_px":8,"spike_detour_ratio":2})
        self.assertGreater(n,0)
        self.assertNotIn([4.,4.],cleaned.tolist())
        self.assertEqual(tuple(cleaned[0]),(0,10))
        self.assertEqual(tuple(cleaned[-1]),(12,10))

    def test_bay_headland_not_flattened(self):
        points=[(0,15),(10,15),(17,5),(23,2),(30,15),(40,15)]
        cleaned, n=remove_spikes(points, {
            "spike_span_px":7,"spike_detour_ratio":2.8})
        self.assertEqual(n,0)
        self.assertEqual(len(cleaned),len(points))

    def test_natural_versus_built_context(self):
        image=np.zeros((100,180,3),np.uint8)
        image[:,:90]=(80,155,90)
        image[:,90:]=(155,170,185)
        water=np.zeros((100,180),np.uint8)
        opts={"natural_hsv_ranges":[{"lower":[20,25,30],"upper":[90,255,255]}],
              "built_hsv_ranges":[{"lower":[0,0,30],"upper":[20,150,255]}],
              "context_radius_px":5}
        land=LineString([(35,20),(35,70)])
        city=LineString([(125,20),(125,70)])
        a,_,_=segment_evidence(land,image,{},opts,water)
        b,_,_=segment_evidence(city,image,{},opts,water)
        self.assertEqual(a,"natural_candidate")
        self.assertEqual(b,"built_candidate")


if __name__=="__main__":
    unittest.main()
