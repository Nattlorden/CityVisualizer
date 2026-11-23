"""Synthetic v0.24 regression tests: thickness, darkness and real graph continuity."""
import unittest

import cv2
import numpy as np

from wall_trace import thin_masonry
from wall_trace_thickness import (evidence_fields, graph_neighbors,
                                  trace_weighted)


class ThicknessAwareWallTests(unittest.TestCase):
    def config(self):
        return {
            "target_halfwidth_px": 3,
            "darkness_scale_px": 80,
            "house_penalty_radius_px": 3,
            "thick_wall_house_relief": .70,
            "base_evidence": .12,
            "thickness_weight": .55,
            "darkness_weight": .33,
            "house_penalty_weight": .20,
            "minimum_step_evidence": .32,
            "anchor_reach_px": 7,
            "max_trace_steps": 300,
            "min_wall_trace_length_px": 12,
            "max_starts_per_tower": 16,
            "start_direction_bins": 16,
            "junction_score_margin": .10,
        }

    def test_thick_black_wall_scores_above_gray_road(self):
        gray = np.full((110, 190), 220, np.uint8)
        black = np.zeros_like(gray)
        roofs = np.zeros_like(gray)
        cv2.line(gray, (10, 26), (170, 26), 18, 7)
        cv2.line(gray, (10, 78), (170, 78), 65, 2)
        cv2.line(black, (10, 26), (170, 26), 255, 7)
        cv2.line(black, (10, 78), (170, 78), 255, 2)
        result = evidence_fields(gray, black, roofs, self.config())
        self.assertGreater(result["wall_score"][26, 85],
                           result["wall_score"][78, 85] + .25)
        self.assertGreater(result["thickness"][26, 85],
                           result["thickness"][78, 85])

    def test_roof_penalty_never_removes_black_wall(self):
        gray = np.full((100, 150), 220, np.uint8)
        black = np.zeros_like(gray)
        cv2.line(black, (12, 40), (135, 40), 255, 7)
        cv2.line(gray, (12, 40), (135, 40), 18, 7)
        roofs = np.zeros_like(gray)
        roofs[30:58, 72:103] = 255
        a = evidence_fields(gray, black, np.zeros_like(roofs),
                            self.config())
        b = evidence_fields(gray, black, roofs, self.config())
        self.assertLess(b["wall_score"][40, 80],
                        a["wall_score"][40, 80])
        self.assertGreater(b["wall_score"][40, 80], .55)
        self.assertEqual(np.count_nonzero(black), np.count_nonzero(black))

    def test_redundant_diagonal_not_a_second_branch(self):
        skeleton = np.zeros((35, 35), dtype=bool)
        skeleton[15, 15] = True
        skeleton[15, 16] = True
        skeleton[16, 16] = True
        neighbors = graph_neighbors((15, 15), skeleton)
        self.assertIn((16, 15), neighbors)
        self.assertNotIn((16, 16), neighbors)

    def test_no_bridge_between_disconnected_towers(self):
        black = np.zeros((120, 220), np.uint8)
        gray = np.full_like(black, 220)
        anchors = np.zeros_like(black)
        anchors[54:67, 12:25] = 255
        anchors[54:67, 160:173] = 255
        black[anchors > 0] = 255
        gray[anchors > 0] = 15
        cv2.line(black, (24, 60), (96, 60), 255, 5)
        cv2.line(gray, (24, 60), (96, 60), 15, 5)
        fields = evidence_fields(gray, black, np.zeros_like(black),
                                 self.config())
        skeleton = thin_masonry(black)
        routes, _ = trace_weighted(skeleton, anchors, fields,
                                   self.config())
        for item in routes:
            xs = [p[0] for p in item["line"].coords]
            self.assertFalse(100 < min(xs) < 158)
            self.assertFalse(100 < max(xs) < 158)

    def test_scores_remain_bounded_and_zero_off_ink(self):
        gray = np.full((30, 30), 120, np.uint8)
        black = np.zeros_like(gray)
        black[10:20, 10:20] = 255
        fields = evidence_fields(gray, black, np.zeros_like(black),
                                 self.config())
        score = fields["wall_score"]
        self.assertEqual(float(score[0, 0]), 0)
        self.assertTrue(np.isfinite(score).all())
        self.assertLessEqual(float(score.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
