"""v0.18: matching-stage geometry, with v0.16 candidate as A/B reference.

Reject unsupported *transitions in the dynamic-programming matcher* rather
than deleting geometries after simplification. Quarantine small closed water
contours as review-only; preserve the water and the previous version.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import mapping

from coast_guided import (coast_contours, normals_toward_land,
                          candidate_samples, paths_to_lines)
from coast_refine import refine, remove_spikes
from extract import read_image, save_png
from land_boundary_dual import ink_mask
from shoreline import v07_mask
from water import spatial_mask


def transition_support(start, stop, ink_near):
    distance = float(np.linalg.norm(np.asarray(stop, float) -
                                    np.asarray(start, float)))
    n = max(2, int(math.ceil(distance * 2)) + 1)
    pts = np.rint(np.linspace(start, stop, n)).astype(int)
    h, w = ink_near.shape
    if np.any((pts[:, 0] < 0) | (pts[:, 1] < 0) |
              (pts[:, 0] >= w) | (pts[:, 1] >= h)):
        return 0.0
    return float(np.mean(ink_near[pts[:, 1], pts[:, 0]]))


def choose_spatial_path(rows, ink, config, match):
    """Viterbi-like coast matching with explicit gap and geometric constraints.

    Constraints apply BEFORE accepting a connection. A supported long straight
    quay is permitted; a similarly long diagonal across blank map is not.
    """
    if not rows:
        return [], {"rejected_long": 0, "rejected_unsupported": 0}
    radius = int(match.get("ink_support_radius_px", 2))
    if not 0 <= radius <= 6:
        raise ValueError("ink_support_radius_px must be 0..6")
    ink_near = (cv2.dilate(ink, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))) > 0
        if radius else ink > 0)
    max_distance = float(match.get("max_point_step_px", 35))
    check_from = float(match.get("verify_step_from_px", 10))
    min_fraction = float(match.get("min_ink_link_fraction", 0.28))
    max_offset_change = float(config.get("max_step_offset_change_px", 18))
    states = [[None] + row for row in rows]
    scores = []
    back = []
    diagnostics = {"rejected_long": 0, "rejected_unsupported": 0}
    for index, row in enumerate(states):
        cost = np.full(len(row), np.inf)
        parent = np.full(len(row), -1, dtype=int)
        for j, current in enumerate(row):
            local = (float(config.get("gap_penalty", 1.65))
                     if current is None else
                     float(config.get("distance_penalty", .025))*current[0] -
                     float(config.get("ink_reward", 2.6)))
            if index == 0:
                cost[j] = local
                continue
            for k, prev in enumerate(states[index-1]):
                if not np.isfinite(scores[-1][k]):
                    continue
                if current is None and prev is None:
                    delta = 0.0
                elif current is None or prev is None:
                    delta = float(config.get("gap_penalty", 1.65))*0.3
                else:
                    change = abs(current[0]-prev[0])
                    if change > max_offset_change:
                        continue
                    distance = float(np.linalg.norm(
                        np.asarray(current[1], float) -
                        np.asarray(prev[1], float)))
                    if distance > max_distance:
                        diagnostics["rejected_long"] += 1
                        continue
                    if distance > check_from and transition_support(
                            prev[1], current[1], ink_near) < min_fraction:
                        diagnostics["rejected_unsupported"] += 1
                        continue
                    delta = float(config.get("distance_jump_penalty", .13))*change
                value = scores[-1][k] + local + delta
                if value < cost[j]:
                    cost[j], parent[j] = value, k
        scores.append(cost)
        back.append(parent)
    result = [None]*len(states)
    position = int(np.argmin(scores[-1]))
    for i in range(len(states)-1, -1, -1):
        result[i] = states[i][position]
        if i:
            position = int(back[i][position])
            if position < 0:
                raise RuntimeError("Matching unexpectedly lost all valid predecessors")
    return result, diagnostics


def small_closed_contour(contour, options):
    """Only review small CLOSED exterior water bodies; never infer islands."""
    if contour["water_hole"] or not contour["closed"]:
        return False
    pts = contour["points"]
    if len(pts) < 5:
        return False
    x, y = pts[:, 0], pts[:, 1]
    area = abs(float(np.dot(x, np.roll(y, -1)) -
                     np.dot(y, np.roll(x, -1))))/2.0
    perimeter = float(np.linalg.norm(pts-np.roll(pts, -1, axis=0),
                                     axis=1).sum())
    if perimeter <= 0:
        return False
    circularity = 4*math.pi*area/(perimeter*perimeter)
    return (float(options.get("small_water_min_area_px2", 30))
            <= area <= float(options.get("small_water_max_area_px2", 2200))
            and circularity >= float(options.get("small_water_min_circularity", .32)))


def match_contours(original, hdr, profile, matching, water=None):
    baseline = dict(profile["coast_guided_v015"])
    if original.shape != hdr.shape:
        raise ValueError("Source and HDR must be pixel-aligned")
    water = v07_mask(original, profile) if water is None else water.copy()
    allowed = spatial_mask(original.shape, profile)
    ink = ink_mask(hdr, profile["land_boundary_v012"]) & allowed
    contours = coast_contours(water, allowed, baseline)
    accepted, review = [], []
    diagnostics = {"contour_runs":len(contours), "small_water_review":0,
                   "rejected_long":0,"rejected_unsupported":0,
                   "unmatched_samples":0}
    selected_hits = np.zeros_like(water)
    review_hits = np.zeros_like(water)
    for index, contour in enumerate(contours):
        pts = contour["points"]
        normals = normals_toward_land(
            pts, water, baseline, contour["closed"])
        candidates = candidate_samples(pts, normals, ink, allowed, baseline)
        chosen, stats = choose_spatial_path(candidates, ink, baseline, matching)
        diagnostics["unmatched_samples"] += sum(c is None for c in chosen)
        diagnostics["rejected_long"] += stats["rejected_long"]
        diagnostics["rejected_unsupported"] += stats["rejected_unsupported"]
        suspicious = small_closed_contour(contour, matching)
        if suspicious:
            diagnostics["small_water_review"] += 1
        target = review if suspicious else accepted
        mask = review_hits if suspicious else selected_hits
        for value in chosen:
            if value is not None:
                x,y = value[1]
                mask[y,x] = 255
        for line in paths_to_lines(chosen, baseline):
            coords, _ = remove_spikes(
                line.coords, profile["coast_refine_v016"])
            if len(coords) < 2:
                continue
            from shapely.geometry import LineString
            cleaned = LineString(coords)
            if cleaned.length < float(baseline.get("min_segment_length_px",14)):
                continue
            target.append({
                "line":cleaned, "source_contour":index,
                "water_hole":contour["water_hole"],
                "review_reason": "small_closed_water_no_proven_beach"
                                 if suspicious else None})
    return water, accepted, review, selected_hits, review_hits, diagnostics


def export_features(records, prefix, review):
    features = []
    for i, item in enumerate(records, 1):
        fid = f"{prefix}{i:04d}"
        features.append({"type":"Feature","id":fid,
            "properties":{"id":fid,"layer":"inner_land_boundary",
                "classification":"candidate",
                "review_status":"needs_review" if review else "unreviewed",
                "reason":item["review_reason"],
                "source_water_contour":item["source_contour"],
                "boundary_role":"water_hole" if item["water_hole"]
                                 else "water_exterior",
                "coordinate_system":"image_pixels_top_left_origin"},
            "geometry":mapping(item["line"])})
    return {"type":"FeatureCollection","features":features}


def main():
    p=argparse.ArgumentParser(description="v0.18 coast matcher geometry")
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--hdr",type=Path,required=True)
    p.add_argument("--profile",type=Path,required=True)
    p.add_argument("--water-mask",type=Path)
    p.add_argument("--output",type=Path,default=Path("output/coast_matching_v018"))
    args=p.parse_args()
    original,hdr=read_image(args.image),read_image(args.hdr)
    profile=json.loads(args.profile.read_text(encoding="utf-8"))
    options=profile["coast_matching_v018"]
    override=None
    if args.water_mask:
        override=cv2.imread(str(args.water_mask),cv2.IMREAD_GRAYSCALE)
        if override is None or override.shape != original.shape[:2]:
            raise ValueError("Invalid v0.7 water mask")
        override=np.where(override>127,255,0).astype(np.uint8)
    water, accepted, review, hits, review_hits, stats = match_contours(
        original,hdr,profile,options,override)
    _, old, _, _=refine(
        original,hdr,profile,profile["coast_refine_v016"],override)
    folder=args.output
    folder.mkdir(parents=True,exist_ok=True)
    boundary=cv2.morphologyEx(water,cv2.MORPH_GRADIENT,
                             np.ones((3,3),np.uint8))
    reference=hdr.copy()
    result=hdr.copy()
    reference[boundary>0]=(240,130,20)
    result[boundary>0]=(240,130,20)
    for item in old:
        coords=np.rint(np.asarray(item["line"].coords)).astype(np.int32)
        cv2.polylines(reference,[coords],False,(0,0,235),1)
    for item in accepted:
        coords=np.rint(np.asarray(item["line"].coords)).astype(np.int32)
        cv2.polylines(result,[coords],False,(0,0,235),1)
    for item in review:
        coords=np.rint(np.asarray(item["line"].coords)).astype(np.int32)
        cv2.polylines(result,[coords],False,(0,200,245),2)
    save_png(folder/"hdr_reference_v016.png",reference)
    save_png(folder/"hdr_matching_v018.png",result)
    save_png(folder/"matched_hits_v018.png",hits)
    save_png(folder/"small_water_review_hits_v018.png",review_hits)
    save_png(folder/"water_v07_unchanged.png",water)
    for name,objects,is_review in (
        ("coast_candidates_v018.geojson",accepted,False),
        ("small_water_review_v018.geojson",review,True)):
        (folder/name).write_text(
            json.dumps(export_features(objects,"RV" if is_review else "LB",
                                       is_review),indent=2),encoding="utf-8")
    stats.update({"version":"0.18","reference_v016_segments":len(old),
                  "candidate_segments":len(accepted),
                  "review_segments":len(review),
                  "note":"Unreviewed matching; neither v0.7 nor previous versions changed."})
    (folder/"stats_v018.json").write_text(json.dumps(stats,indent=2),
                                          encoding="utf-8")
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
