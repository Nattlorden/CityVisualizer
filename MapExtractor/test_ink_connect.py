"""Synthetic regression tests for v0.13 ink connectivity."""
import unittest
import cv2
import numpy as np
from ink_connect import skeletonize, propose_bridges, trace_chains, candidate_components

class InkConnectTests(unittest.TestCase):
    def test_facing_gap_connects(self):
        mask=np.zeros((80,120),np.uint8)
        cv2.line(mask,(10,30),(49,30),255,1)
        cv2.line(mask,(56,30),(110,30),255,1)
        thin=skeletonize(mask)
        bridge, links, _=propose_bridges(thin,np.ones(mask.shape,bool),
            {"max_gap_px":10,"direction_radius_px":5,"min_direction_cos":0.6})
        self.assertEqual(len(links),1)
        self.assertGreater(np.count_nonzero(bridge[30,50:56]),0)

    def test_perpendicular_ends_not_joined(self):
        mask=np.zeros((80,120),np.uint8)
        cv2.line(mask,(10,30),(49,30),255,1)
        cv2.line(mask,(55,35),(55,65),255,1)
        thin=skeletonize(mask)
        _, links, _=propose_bridges(thin,np.ones(mask.shape,bool),
            {"max_gap_px":12,"direction_radius_px":5,"min_direction_cos":0.7})
        self.assertEqual(len(links),0)

    def test_closed_island_loop_is_traced(self):
        mask=np.zeros((90,90),np.uint8)
        cv2.rectangle(mask,(20,20),(60,60),255,1)
        chains=trace_chains(skeletonize(mask),min_length=10)
        self.assertGreaterEqual(len(chains),1)

    def test_coastal_selection_does_not_destroy_source(self):
        mask=np.zeros((80,120),np.uint8)
        cv2.line(mask,(10,15),(70,15),255,1)
        cv2.line(mask,(10,65),(70,65),255,1)
        corridor=np.zeros(mask.shape,bool)
        corridor[:40]=True
        chosen, report=candidate_components(mask,corridor,
            {"min_component_pixels":15,"min_corridor_fraction":0.6})
        self.assertGreater(np.count_nonzero(chosen[15]),0)
        self.assertEqual(np.count_nonzero(chosen[65]),0)
        self.assertEqual(len(report),2)

if __name__=="__main__":
    unittest.main()
