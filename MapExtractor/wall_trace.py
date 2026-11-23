"""v0.22: black-ink wall tracing from tower anchors, with NO forced enclosure.

Strong black wall pixels are traced outwards from v0.21 tower candidates.
The algorithm stops at ambiguous junctions/dead ends: no guessed tower links,
no graph closure, and no gates inferred from unrelated gaps. Every export is
an unreviewed image-pixel candidate.
"""
import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, mapping

from extract import read_image, save_png
from fortifications import analyze as fortification_baseline, ellipsoid
from shoreline import v07_mask


N8 = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]


def darkness_layers(hdr, cfg, region):
    """Absolute dark core plus near-black gray-scale diagnostics."""
    gray = cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)
    # The LAB lightness display makes black masonry vs gray roads inspectable.
    lab = cv2.cvtColor(hdr, cv2.COLOR_BGR2LAB)
    threshold = int(cfg.get("black_gray_max", 75))
    if not 5 <= threshold <= 180:
        raise ValueError("black_gray_max outside 5..180")
    near_black = np.where((gray <= threshold) & (region > 0),
                          255, 0).astype(np.uint8)
    # Suppress isolated printing specks but retain slender masonry.
    opening = int(cfg.get("black_open_radius", 0))
    if opening:
        near_black = cv2.morphologyEx(near_black, cv2.MORPH_OPEN,
                                      ellipsoid(opening))
    return {"gray_value": gray, "lab_lightness": lab[:, :, 0],
            "near_black": near_black}


def tower_seeds(tower_mask, black, cfg, water=None):
    """Retain the existing v0.21 tower polygons and select strong anchors.

    Perimeter of a tower is an allowed starting point; walls and towers are
    not assumed to be topologically connected to any other tower.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (tower_mask > 0).astype(np.uint8), connectivity=8)
    minimum = int(cfg.get("min_tower_area_px", 20))
    max_area = int(cfg.get("max_tower_area_px", 3000))
    support_min = float(cfg.get("min_tower_black_fraction", 0.55))
    near_water = None
    if water is not None and cfg.get("tower_water_proximity_px") is not None:
        reach = float(cfg["tower_water_proximity_px"])
        if reach > 0:
            near_water = cv2.distanceTransform(
                (water == 0).astype(np.uint8), cv2.DIST_L2, 3) <= reach
    seed_mask = np.zeros_like(black)
    rejected = []
    seeds = []
    for i in range(1, count):
        pixels = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not minimum <= area <= max_area:
            rejected.append({"source_component": i, "reason": "tower_size"})
            continue
        support = float(np.mean(black[pixels] > 0))
        if support < support_min:
            rejected.append({"source_component": i, "reason": "not_black_enough",
                             "fraction": round(support, 3)})
            continue
        if near_water is not None and not np.any(pixels & near_water):
            rejected.append({"source_component": i, "reason": "not_near_water"})
            continue
        # Retain the full tower footprint, not merely its central black core.
        seed_mask[pixels] = 255
        seeds.append({"tower_id": f"T{i:04d}", "source_component": i,
                      "area_px": area, "black_fraction": round(support,3)})
    return seed_mask, seeds, rejected


def thin_masonry(black):
    """OpenCV-only skeleton; no ximgproc dependency."""
    img = (black > 0).astype(np.uint8)*255
    result = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img):
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        result |= cv2.subtract(img, opened)
        if cv2.countNonZero(eroded) == cv2.countNonZero(img):
            # Defensive bail-out for unexpected morphological stagnation.
            result |= img
            break
        img = eroded
    return result


def degree_map(skeleton):
    kernel = np.ones((3, 3), np.uint8)
    kernel[1,1] = 0
    return cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel,
                        borderType=cv2.BORDER_CONSTANT)


def trace_tower_arms(skeleton, seed_mask, cfg):
    """Trace dark skeleton segments beginning near a tower.

    Stop at junctions, other towers and ends; never infer a missing link.
    Multiple arms leaving a tower remain separate candidate wall edges.
    """
    h,w = skeleton.shape
    skeleton_bool = skeleton > 0
    degree = degree_map(skeleton)
    tower_near = cv2.dilate(seed_mask, ellipsoid(
        int(cfg.get("anchor_reach_px", 5)))) > 0
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        (seed_mask > 0).astype(np.uint8), connectivity=8)
    # Do not regard skeleton pixels *inside* tower bodies as wall length.
    exposed = skeleton_bool & (seed_mask == 0)
    starting_points = []
    for y,x in np.argwhere(exposed & tower_near):
        adjacent = labels[max(0,y-4):min(h,y+5), max(0,x-4):min(w,x+5)]
        positive = adjacent[adjacent > 0]
        if len(positive):
            first = int(np.bincount(positive).argmax())
            starting_points.append((x,y,first))
    starting_points.sort()
    max_steps = int(cfg.get("max_trace_steps", 650))
    minimum = float(cfg.get("min_wall_trace_length_px", 17))
    visited_edges = set()
    routes = []
    def key(a,b): return (a,b) if a < b else (b,a)
    for x,y,origin in starting_points:
        start = (int(x),int(y))
        # Skip interior branch clutter. Start from nodes with a free continuation.
        neighbors = [(x+dx,y+dy) for dx,dy in N8
                     if 0 <= x+dx < w and 0 <= y+dy < h
                     and exposed[y+dy,x+dx]]
        if not neighbors:
            continue
        for nxt in neighbors:
            if key(start,nxt) in visited_edges:
                continue
            chain = [start]
            current = start
            previous = None
            end_reason = "max_trace"
            for _ in range(max_steps):
                choices = [(current[0]+dx,current[1]+dy) for dx,dy in N8
                           if 0 <= current[0]+dx < w and 0 <= current[1]+dy < h
                           and exposed[current[1]+dy,current[0]+dx]
                           and (current[0]+dx,current[1]+dy) != previous]
                if previous is None:
                    choices = [nxt]
                if not choices:
                    end_reason = "dead_end"
                    break
                if len(choices) > 1:
                    # Diagonal 8-connectivity creates spurious forks along
                    # ordinary one-pixel curves. Continue only when exactly
                    # one branch follows the incoming direction much better.
                    if previous is None:
                        end_reason = "junction_review"
                        break
                    incoming = np.asarray(current, float) - np.asarray(previous, float)
                    incoming /= max(float(np.linalg.norm(incoming)), 1e-6)
                    alignments = sorted(
                        ((float(np.dot(
                            (np.asarray(point, float)-np.asarray(current, float)) /
                            max(float(np.linalg.norm(
                                np.asarray(point, float)-np.asarray(current, float))),1e-6),
                            incoming)), point) for point in choices),
                        reverse=True)
                    threshold = float(cfg.get("min_forward_alignment", 0.35))
                    margin = float(cfg.get("branch_alignment_margin", 0.22))
                    if (alignments[0][0] < threshold or
                            alignments[0][0] - alignments[1][0] < margin):
                        end_reason = "junction_review"
                        break
                    next_pt = alignments[0][1]
                else:
                    next_pt = choices[0]
                e = key(current,next_pt)
                if e in visited_edges:
                    end_reason = "existing_trace"
                    break
                visited_edges.add(e)
                chain.append(next_pt)
                previous,current = current,next_pt
                if degree[current[1],current[0]] >= int(
                        cfg.get("branch_degree_stop", 4)):
                    end_reason = "junction_review"
                    break
                if tower_near[current[1],current[0]] and len(chain)>3:
                    # Only a confirmed different source tower terminates as
                    # tower-to-tower; proximity is not evidence of a link.
                    zone = labels[max(0,current[1]-4):min(h,current[1]+5),
                                  max(0,current[0]-4):min(w,current[0]+5)]
                    other = set(int(v) for v in np.unique(zone) if v>0)
                    if any(v != origin for v in other):
                        end_reason = "other_tower"
                        break
            if len(chain) < 2:
                continue
            line = LineString(chain)
            if line.length >= minimum:
                routes.append({"line":line, "origin_component":origin,
                               "end_reason":end_reason})
    return routes


def main():
    p = argparse.ArgumentParser(description="v0.22 tower-anchored black wall tracing")
    p.add_argument("--image", type=Path, required=True,
                   help="HDR image; do not use the colored candidate overlay")
    p.add_argument("--original", type=Path, help="Original photo, for optional water reference")
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--building-config", type=Path,
                   default=Path("MapExtractor/sample-config.json"))
    p.add_argument("--output", type=Path,
                   default=Path("output/fortifications_v022"))
    args = p.parse_args()
    hdr = read_image(args.image)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    buildings = json.loads(args.building_config.read_text(encoding="utf-8"))
    cfg = profile["fortifications_v022"]
    prev = fortification_baseline(hdr, profile, buildings,
                                 profile["fortifications_v021"])
    layers = darkness_layers(hdr, cfg, prev["analysis_region"])
    water = None
    if args.original is not None:
        original = read_image(args.original)
        if original.shape != hdr.shape:
            raise ValueError("Original/HDR sizes differ")
        water = v07_mask(original,profile)
    seeds, anchors, rejected = tower_seeds(
        prev["tower_thickenings"],layers["near_black"],cfg,water)
    # Suppress paths within entire thick tower regions before tracing.
    skeleton = thin_masonry(layers["near_black"])
    routes = trace_tower_arms(skeleton,seeds,cfg)
    folder = args.output; folder.mkdir(parents=True,exist_ok=True)
    for name,mask in layers.items():
        save_png(folder/(name+".png"),mask)
    save_png(folder/"tower_anchors.png",seeds)
    save_png(folder/"black_skeleton.png",skeleton)
    diagnostics = hdr.copy()
    diagnostics[layers["near_black"]>0]=(255,145,45)
    diagnostics[seeds>0]=(205,35,215)
    traced = hdr.copy()
    traced[seeds>0]=(205,35,215)
    features = []
    for i,route in enumerate(routes,1):
        fid=f"W{i:04d}"
        color = (30,200,30) if route["end_reason"]=="other_tower" else (0,200,255)
        pts=np.rint(np.asarray(route["line"].coords)).astype(np.int32)
        cv2.polylines(traced,[pts],False,color,2)
        features.append({"type":"Feature","id":fid,
            "properties":{"id":fid,"classification":"wall_trace_candidate",
                "review_status":"unreviewed", "source_tower_component":route["origin_component"],
                "termination":route["end_reason"],
                "coordinate_system":"image_pixels_top_left_origin"},
            "geometry":mapping(route["line"])})
    save_png(folder/"black_vs_gray_overlay.png",diagnostics)
    save_png(folder/"tower_guided_wall_overlay.png",traced)
    (folder/"wall_traces_v022.geojson").write_text(json.dumps({
        "type":"FeatureCollection","features":features},indent=2),
        encoding="utf-8")
    summary={"version":"0.22","tower_anchors":len(anchors),
             "rejected_tower_components":len(rejected),
             "wall_trace_fragments":len(routes),
             "terminations":{reason:sum(x["end_reason"]==reason for x in routes)
                 for reason in sorted(set(x["end_reason"] for x in routes))},
             "note":"No forced graph closure and no port classification. "
                    "All fragments remain unreviewed."}
    (folder/"stats_v022.json").write_text(json.dumps({
        **summary,"anchors":anchors,"rejected_anchors":rejected},indent=2),
        encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
