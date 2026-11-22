"""v0.15: match landward HDR ink to v0.7 water-contour normals.

Produces unreviewed inner-land-boundary segments; never changes v0.7 water.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, mapping

from extract import read_image, save_png
from land_boundary_dual import ink_mask
from shoreline import v07_mask
from water import spatial_mask


def coast_contours(water, permitted, cfg):
    contours, hierarchy = cv2.findContours(water, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    inside = cv2.distanceTransform(
        np.pad((permitted > 0).astype(np.uint8), 1),
        cv2.DIST_L2, 3)[1:-1, 1:-1]
    step = max(1, int(cfg.get("sample_step_px", 4)))
    result = []
    for index, contour in enumerate(contours):
        full = contour.reshape(-1, 2)
        if len(full) < int(cfg.get("min_contour_points", 40)):
            continue
        pts = full[::step]
        if len(pts) < 5:
            continue
        good = inside[pts[:, 1], pts[:, 0]] > float(
            cfg.get("map_edge_clearance_px", 5))
        if not good.all():
            start = int(np.flatnonzero(~good)[0])
            pts, good = np.roll(pts, -start, axis=0), np.roll(good, -start)
        if not good.any():
            continue
        runs, start = [], None
        for j, ok in enumerate(good):
            if ok and start is None:
                start = j
            if not ok and start is not None:
                runs.append(pts[start:j]); start = None
        if start is not None:
            runs.append(pts[start:])
        for run in runs:
            if len(run) >= 5:
                result.append({"points": run.astype(float),
                    "water_hole": bool(hierarchy[0][index][3] >= 0),
                    "closed": bool(good.all())})
    return result


def normals_toward_land(points, water, cfg, closed=False):
    h, w = water.shape
    k = min(max(2, int(cfg.get("tangent_half_window", 3))),
            max(2, len(points)//5))
    normals = np.zeros_like(points)
    probe = float(cfg.get("normal_probe_px", 5))
    for i, p in enumerate(points):
        ia = (i-k) % len(points) if closed else max(i-k, 0)
        ib = (i+k) % len(points) if closed else min(i+k, len(points)-1)
        tangent = points[ib]-points[ia]
        magnitude = float(np.linalg.norm(tangent))
        if magnitude < 1e-4:
            continue
        n = np.array([-tangent[1], tangent[0]]) / magnitude
        def water_value(sign):
            q = np.rint(p + sign*n*probe).astype(int)
            if q[0] < 0 or q[1] < 0 or q[0] >= w or q[1] >= h:
                return -1
            return int(water[q[1], q[0]] > 0)
        a, b = water_value(1), water_value(-1)
        if a == 0 and b == 1:
            normals[i] = n
        elif b == 0 and a == 1:
            normals[i] = -n
    return normals


def candidate_samples(points, normals, ink, allowed, cfg):
    h, w = ink.shape
    first = int(cfg.get("min_offset_px", 4))
    last = int(cfg.get("max_offset_px", 75))
    if first < 1 or last < first or last > 240:
        raise ValueError("Invalid landward offset range")
    radius = max(0, int(cfg.get("ink_search_radius_px", 1)))
    lookup = ink if radius == 0 else cv2.dilate(ink,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2*radius+1, 2*radius+1)))
    rows = []
    for point, normal in zip(points, normals):
        candidates = []
        if float(np.linalg.norm(normal)) < 0.5:
            rows.append(candidates)
            continue
        last_hit = None
        for d in range(first, last+1):
            q = np.rint(point + normal*d).astype(int)
            x, y = int(q[0]), int(q[1])
            if x < 0 or y < 0 or x >= w or y >= h or not allowed[y, x]:
                break
            if lookup[y, x] == 0:
                last_hit = None
                continue
            if last_hit is None or d-last_hit > 1:
                candidates.append((float(d), (x, y)))
            last_hit = d
        rows.append(candidates)
    return rows


def choose_path(candidates, cfg):
    """Choose mutually continuous ink hits; missing hits remain explicit gaps."""
    if not candidates:
        return []
    jump = float(cfg.get("distance_jump_penalty", 0.13))
    distance_cost = float(cfg.get("distance_penalty", 0.025))
    gap_cost = float(cfg.get("gap_penalty", 1.65))
    ink_reward = float(cfg.get("ink_reward", 2.6))
    max_jump = float(cfg.get("max_step_offset_change_px", 18))
    states = [[None] + row for row in candidates]
    scores, parents = [], []
    for i, row in enumerate(states):
        current = np.full(len(row), np.inf)
        parent = np.full(len(row), -1, dtype=int)
        for j, state in enumerate(row):
            local = gap_cost if state is None else distance_cost*state[0]-ink_reward
            if i == 0:
                current[j] = local
                continue
            for k, prev in enumerate(states[i-1]):
                if state is None and prev is None:
                    transition = 0.0
                elif state is None or prev is None:
                    transition = gap_cost*0.3
                else:
                    difference = abs(state[0]-prev[0])
                    if difference > max_jump:
                        continue
                    transition = jump*difference
                trial = scores[-1][k] + local + transition
                if trial < current[j]:
                    current[j], parent[j] = trial, k
        scores.append(current)
        parents.append(parent)
    chosen = [None]*len(states)
    index = int(np.argmin(scores[-1]))
    for i in range(len(states)-1, -1, -1):
        chosen[i] = states[i][index]
        index = int(parents[i][index]) if i else 0
        if index < 0 and i:
            break
    return chosen


def paths_to_lines(chosen, cfg):
    minimum = int(cfg.get("min_supported_samples", 8))
    lines, points = [], []
    def flush():
        if len(points) < minimum:
            return
        coords = [p[1] for p in points]
        if len(set(coords)) < 2:
            return
        line = LineString(coords).simplify(
            float(cfg.get("simplify_px", 0.8)), preserve_topology=True)
        if line.length >= float(cfg.get("min_segment_length_px", 14)):
            lines.append(line)
    for state in chosen:
        if state is None:
            flush(); points = []
        else:
            points.append(state)
    flush()
    return lines


def run(image, hdr, profile, cfg, water_override=None):
    if image.shape != hdr.shape:
        raise ValueError("Original and HDR must have matching image dimensions")
    water = v07_mask(image, profile) if water_override is None else water_override.copy()
    allowed = spatial_mask(image.shape, profile)
    ink = ink_mask(hdr, profile["land_boundary_v012"]) & allowed
    contours = coast_contours(water, allowed, cfg)
    lines = []
    chosen_pixels = np.zeros_like(water)
    rays = np.zeros_like(water)
    unknown = 0
    for contour in contours:
        points = contour["points"]
        normals = normals_toward_land(points, water, cfg, contour["closed"])
        options = candidate_samples(points, normals, ink, allowed, cfg)
        chosen = choose_path(options, cfg)
        unknown += sum(item is None for item in chosen)
        for item in chosen:
            if item is not None:
                x,y = item[1]
                chosen_pixels[y,x] = 255
        for point, normal in zip(points[::4], normals[::4]):
            if np.linalg.norm(normal) > 0.5:
                end = point+normal*int(cfg.get("max_offset_px",75))
                cv2.line(rays, tuple(np.rint(point).astype(int)),
                         tuple(np.rint(end).astype(int)),255,1)
        for line in paths_to_lines(chosen, cfg):
            lines.append({"line":line, "water_hole":contour["water_hole"]})
    masks = {"water_v07":water, "ink_hdr_raw":ink,
             "sampled_rays":rays, "selected_ink_hits":chosen_pixels}
    return masks, lines, {"water_contour_runs":len(contours),
                           "unmatched_samples":unknown}


def main():
    parser = argparse.ArgumentParser(description="v0.15 ink matching along water normals")
    parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--hdr",type=Path,required=True)
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--water-mask",type=Path,
                        help="Optional externally saved v0.7 mask")
    parser.add_argument("--output",type=Path,default=Path("output/coast_guided_v015"))
    args=parser.parse_args()
    image,hdr=read_image(args.image),read_image(args.hdr)
    profile=json.loads(args.profile.read_text(encoding="utf-8"))
    cfg=profile.get("coast_guided_v015")
    if cfg is None:
        raise ValueError("Missing coast_guided_v015 profile section")
    override=None
    if args.water_mask:
        override=cv2.imread(str(args.water_mask), cv2.IMREAD_GRAYSCALE)
        if override is None or override.shape != image.shape[:2]:
            raise ValueError("Water mask is missing or has incorrect dimensions")
        override=(override>127).astype(np.uint8)*255
    masks,lines,stats=run(image,hdr,profile,cfg,override)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,mask in masks.items():
        save_png(args.output/(name+".png"), mask)
    features=[]
    for i,obj in enumerate(lines,1):
        fid=f"LB{i:04d}"
        features.append({"type":"Feature","id":fid,
            "properties":{"id":fid,"classification":"inner_land_shore_candidate",
                "boundary_kind":"water_hole" if obj["water_hole"] else "water_exterior",
                "review_status":"unreviewed",
                "coordinate_system":"image_pixels_top_left_origin"},
            "geometry":mapping(obj["line"])})
    for img,name in ((hdr,"hdr_overlay"),(image,"original_overlay")):
        overlay=img.copy()
        edge=cv2.morphologyEx(masks["water_v07"],cv2.MORPH_GRADIENT,
                              np.ones((3,3),np.uint8))
        overlay[edge>0]=(240,130,20)
        overlay[masks["selected_ink_hits"]>0]=(0,200,245)
        for obj in lines:
            coords=np.rint(np.asarray(obj["line"].coords)).astype(np.int32)
            cv2.polylines(overlay,[coords],False,(0,0,255),2)
        save_png(args.output/(name+".png"),overlay)
    (args.output/"land_boundary_v015.geojson").write_text(json.dumps({
        "type":"FeatureCollection","features":features},indent=2),encoding="utf-8")
    stats.update({"version":"0.15","candidate_segments":len(lines),
                  "status":"unreviewed; gaps remain open, v0.7 unchanged"})
    (args.output/"stats_v015.json").write_text(
        json.dumps(stats,indent=2),encoding="utf-8")
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
