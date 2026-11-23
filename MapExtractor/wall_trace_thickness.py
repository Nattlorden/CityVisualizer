"""CityVisualizer v0.24 — thickness-aware, tower-anchored wall tracing.

Keep v0.22/v0.23 unchanged. Skeleton pixels inherit the local width and darkness
of the ORIGINAL black mask. Accepted v0.5 roofs give a soft penalty, never erase
masonry. At forks choose a clearly supported continuation or stop for review.
No invented connection between towers, no presumed closed fortification.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import mapping

from extract import read_image, save_png
from fortifications import analyze as detect_fortifications
from shoreline import v07_mask
from wall_trace import (N8, darkness_layers, thin_masonry, tower_seeds,
                        trace_tower_arms)
from wall_trace_building_aware import (building_roofs_v05, expand_mask,
                                      render, trace_pixels)


def evidence_fields(gray, black, roofs, cfg):
    """Use original stroke half-width, blackness, and soft roof proximity."""
    if gray.shape != black.shape or gray.shape != roofs.shape:
        raise ValueError("Gray, black, and roof masks must have equal dimensions")
    radius = int(cfg.get("house_penalty_radius_px", 3))
    if not 0 <= radius <= 15:
        raise ValueError("house_penalty_radius_px must be between 0 and 15")
    target_width = float(cfg.get("target_halfwidth_px", 3.0))
    dark_scale = float(cfg.get("darkness_scale_px", 80))
    if target_width <= 0 or dark_scale <= 0:
        raise ValueError("target_halfwidth_px and darkness_scale_px must be positive")

    thickness = cv2.distanceTransform(
        (black > 0).astype(np.uint8), cv2.DIST_L2, 3)
    # Small spatial tolerance: the skeletonizer does not always place a pixel
    # on the exact medial axis of thick cartographic line work.
    local_thickness = cv2.dilate(
        thickness, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    width_strength = np.clip(local_thickness / target_width, 0, 1)
    darkness = np.clip((dark_scale - gray.astype(np.float32)) / dark_scale,
                       0, 1)
    roof_near = (expand_mask(roofs, radius) > 0).astype(np.float32)
    # Substantial black thickness reduces — but does not abolish — the
    # penalty from the adjacent roof. All black pixels remain traceable.
    house_penalty = roof_near * (
        1.0 - float(cfg.get("thick_wall_house_relief", .70))*width_strength)
    score = (float(cfg.get("base_evidence", .12)) +
             float(cfg.get("thickness_weight", .55))*width_strength +
             float(cfg.get("darkness_weight", .33))*darkness -
             float(cfg.get("house_penalty_weight", .20))*house_penalty)
    score = np.clip(score, 0, 1).astype(np.float32)
    score[black == 0] = 0
    return {"thickness": thickness, "width_strength": width_strength,
            "darkness_strength": darkness, "house_penalty": house_penalty,
            "wall_score": score}


def graph_neighbors(point, skeleton):
    """8-connected skeleton, excluding redundant diagonal shortcut edges."""
    x, y = point
    h, w = skeleton.shape
    result = []
    for dx, dy in N8:
        xx, yy = x+dx, y+dy
        if not (0 <= xx < w and 0 <= yy < h and skeleton[yy, xx]):
            continue
        # If orthogonal pixels already connect the same two points, their
        # diagonal edge is not a separate branch of the wall graph.
        if dx and dy and (skeleton[y, xx] or skeleton[yy, x]):
            continue
        result.append((xx, yy))
    return result


def segment_key(a, b):
    return (a, b) if a < b else (b, a)


def starting_frontier(skeleton, seeds, evidence, cfg):
    """Up to N outward starting points per tower, never pair towers by distance."""
    h, w = skeleton.shape
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        (seeds > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return [], labels
    reach = int(cfg.get("anchor_reach_px", 5))
    if not 1 <= reach <= 15:
        raise ValueError("anchor_reach_px must be 1..15")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*reach+1, 2*reach+1))
    band = (cv2.dilate(seeds, kernel) > 0) & (seeds == 0) & skeleton
    buckets = defaultdict(list)
    bins = int(cfg.get("start_direction_bins", 16))
    if not 4 <= bins <= 64:
        raise ValueError("start_direction_bins must be 4..64")
    local_radius = reach + 2
    for y, x in np.argwhere(band):
        y, x = int(y), int(x)
        # The closest tower's pixels (not another disconnected roof/line)
        # must lie in the local radius. This is a *starting hypothesis* only.
        ys = slice(max(0, y-local_radius), min(h, y+local_radius+1))
        xs = slice(max(0, x-local_radius), min(w, x+local_radius+1))
        local = labels[ys, xs]
        yy, xx = np.where(local > 0)
        if not len(xx):
            continue
        squared = (xx + xs.start - x)**2 + (yy + ys.start - y)**2
        index = int(np.argmin(squared))
        origin = int(local[yy[index], xx[index]])
        cx, cy = centers[origin]
        angle = math.atan2(y-cy, x-cx)
        sector = int(((angle+math.pi)/(2*math.pi))*bins) % bins
        distance = math.hypot(x-cx, y-cy)
        priority = distance + .5*float(evidence[y, x])
        buckets[(origin, sector)].append((priority, (x, y)))
    per_tower = defaultdict(list)
    for (origin, _), nodes in buckets.items():
        nodes.sort(reverse=True)
        per_tower[origin].append(nodes[0])
    result = []
    max_starts = int(cfg.get("max_starts_per_tower", 16))
    for origin, nodes in sorted(per_tower.items()):
        for _, point in sorted(nodes, reverse=True)[:max_starts]:
            result.append((point, origin, (float(centers[origin][0]),
                                           float(centers[origin][1]))))
    return result, labels


def next_candidates(current, previous, outward, skeleton, score, width,
                    cfg, used_edges):
    """Score real adjacent graph steps; never insert a bridging line."""
    options = []
    if previous is None:
        direction = np.asarray(current, dtype=float) - np.asarray(outward)
    else:
        direction = np.asarray(current, dtype=float) - np.asarray(previous)
    norm = max(float(np.linalg.norm(direction)), 1e-6)
    direction /= norm
    for point in graph_neighbors(current, skeleton):
        if point == previous or segment_key(current, point) in used_edges:
            continue
        x, y = point
        if float(score[y, x]) < float(cfg.get("minimum_step_evidence", .34)):
            continue
        step = np.asarray(point, dtype=float)-np.asarray(current, dtype=float)
        alignment = float(np.dot(direction, step) /
                          max(np.linalg.norm(step), 1e-6))
        if previous is None and alignment < float(
                cfg.get("minimum_initial_outward_alignment", -.25)):
            continue
        if previous is not None and alignment < float(
                cfg.get("minimum_forward_alignment", -.15)):
            continue
        cx, cy = current
        wshift = abs(float(width[y, x]) - float(width[cy, cx]))
        priority = (float(score[y, x]) +
                    float(cfg.get("direction_weight", .24))*alignment -
                    float(cfg.get("width_jump_penalty", .05))*min(wshift, 4))
        options.append((priority, point))
    options.sort(key=lambda item:(-item[0],item[1]))
    return options


def trace_weighted(skeleton, seed_mask, fields, cfg):
    """Trace along existing pixels from a tower, stopping at ambiguous forks."""
    mask = skeleton > 0
    score = fields["wall_score"]
    width = fields["thickness"]
    frontier, tower_labels = starting_frontier(mask, seed_mask, score, cfg)
    h, w = mask.shape
    used = set()
    routes = []
    stops = defaultdict(int)
    minimum_length = float(cfg.get("min_wall_trace_length_px", 17))
    limit = int(cfg.get("max_trace_steps", 650))
    ambiguity = float(cfg.get("junction_score_margin", .10))
    for start, origin, tower_center in frontier:
        if not mask[start[1], start[0]]:
            continue
        line = [start]
        previous = None
        current = start
        locally_used = set()
        reason = "max_trace"
        for _ in range(limit):
            candidates = next_candidates(
                current, previous, tower_center, mask, score, width, cfg,
                used | locally_used)
            if not candidates:
                reason = "dead_end_or_low_evidence"
                break
            if (len(candidates) > 1 and
                    candidates[0][0] - candidates[1][0] < ambiguity):
                reason = "ambiguous_junction"
                break
            new = candidates[0][1]
            e = segment_key(current, new)
            if e in locally_used:
                reason = "repeat_edge"
                break
            locally_used.add(e)
            line.append(new)
            previous, current = current, new
            # An actual pixel-connected trace may end near another tower.
            # This is an observed termination, not an assumed link.
            if len(line) > 4:
                x, y = current
                patch = tower_labels[max(y-2,0):min(y+3,h),
                                     max(x-2,0):min(x+3,w)]
                if np.any((patch > 0) & (patch != origin)):
                    reason = "reached_other_tower"
                    break
        if len(line) < 2:
            continue
        from shapely.geometry import LineString
        geometry = LineString(line)
        if geometry.length < minimum_length:
            continue
        # Reserve edges only for accepted routes, allowing short starts to
        # be retried without stealing the valid route from another tower.
        used.update(locally_used)
        coords = np.asarray(line, dtype=int)
        xs, ys = coords[:, 0], coords[:, 1]
        mean_score = float(np.mean(score[ys, xs]))
        mean_halfwidth = float(np.mean(width[ys, xs]))
        status = ("higher_evidence" if mean_score >= float(
            cfg.get("higher_evidence_threshold", .72))
            else "needs_review")
        routes.append({"line": geometry, "origin_component": origin,
                       "end_reason": reason, "mean_score": round(mean_score, 3),
                       "mean_halfwidth_px": round(mean_halfwidth, 2),
                       "evidence_tier": status})
        stops[reason] += 1
    return routes, {"frontier_points":len(frontier),
                    "termination_counts":dict(stops)}


def heatmap(field, black):
    value = np.clip(field*255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(value, cv2.COLORMAP_TURBO)
    color[black == 0] = (0, 0, 0)
    return color


def write_features(path, routes):
    features = []
    for index, obj in enumerate(routes, 1):
        fid = f"W{index:04d}"
        features.append({
            "type": "Feature", "id": fid,
            "properties": {
                "id": fid, "classification": "wall_trace_candidate",
                "source_tower_component": obj["origin_component"],
                "termination": obj["end_reason"],
                "mean_evidence_score": obj["mean_score"],
                "mean_ink_halfwidth_px": obj["mean_halfwidth_px"],
                "evidence_tier": obj["evidence_tier"],
                "review_status": "unreviewed",
                "coordinate_system": "image_pixels_top_left_origin",
            },
            "geometry": mapping(obj["line"]),
        })
    path.write_text(json.dumps(
        {"type":"FeatureCollection", "features": features},
        indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="v0.24 thickness-aware tower-anchored wall tracer")
    parser.add_argument("--image", type=Path, required=True,
                        help="Unmodified HDR map, not a colored overlay")
    parser.add_argument("--original", type=Path, required=True,
                        help="Original photo for existing v0.5 building candidates")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--building-config", type=Path,
                        default=Path("MapExtractor/sample-config.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/fortifications_v024"))
    args = parser.parse_args()

    hdr, original = read_image(args.image), read_image(args.original)
    if hdr.shape != original.shape:
        raise ValueError("HDR and original images must have identical pixel sizes")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    building_cfg = json.loads(args.building_config.read_text(encoding="utf-8"))
    cfg = profile["fortifications_v024"]
    baseline_cfg = profile["fortifications_v022"]

    previous = detect_fortifications(
        hdr, profile, building_cfg, profile["fortifications_v021"])
    dark = darkness_layers(hdr, baseline_cfg, previous["analysis_region"])
    water = v07_mask(original, profile)
    anchors, anchor_info, excluded = tower_seeds(
        previous["tower_thickenings"], dark["near_black"], baseline_cfg, water)
    roofs, roof_count = building_roofs_v05(original, building_cfg)
    fields = evidence_fields(dark["gray_value"], dark["near_black"],
                             roofs, cfg)

    skeleton = thin_masonry(dark["near_black"])
    baseline = trace_tower_arms(skeleton, anchors, baseline_cfg)
    tracing_cfg = dict(baseline_cfg)
    tracing_cfg.update(cfg)
    weighted, details = trace_weighted(
        skeleton, anchors, fields, tracing_cfg)

    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    save_png(folder/"near_black_v022_unchanged.png", dark["near_black"])
    save_png(folder/"tower_anchors_unchanged.png", anchors)
    save_png(folder/"building_roofs_v05.png", roofs)
    save_png(folder/"thickness_halfwidth_px.png",
             np.uint8(np.clip(fields["thickness"]*20, 0, 255)))
    save_png(folder/"thickness_heatmap.png",
             heatmap(fields["width_strength"], dark["near_black"]))
    save_png(folder/"darkness_strength.png",
             np.uint8(np.clip(fields["darkness_strength"]*255, 0, 255)))
    save_png(folder/"house_penalty.png",
             np.uint8(np.clip(fields["house_penalty"]*255, 0, 255)))
    save_png(folder/"wall_evidence_heatmap.png",
             heatmap(fields["wall_score"], dark["near_black"]))
    save_png(folder/"baseline_v022_wall_overlay.png",
             render(hdr, anchors, baseline))
    overlay = hdr.copy()
    for route in weighted:
        points = np.rint(np.asarray(route["line"].coords)).astype(np.int32)
        color = ((0, 210, 30) if route["evidence_tier"]=="higher_evidence"
                 else (0, 185, 255))
        cv2.polylines(overlay, [points], False, color, 2)
    overlay[anchors > 0] = (205, 35, 215)
    save_png(folder/"thickness_guided_wall_overlay.png", overlay)

    old = trace_pixels(hdr.shape[:2], baseline)>0
    new = trace_pixels(hdr.shape[:2], weighted)>0
    difference = np.zeros_like(hdr)
    difference[old & new] = (165,165,165)
    difference[old & ~new] = (0,0,245)
    difference[~old & new] = (0,215,0)
    save_png(folder/"trace_difference_vs_v022.png",difference)
    write_features(folder/"wall_traces_v024.geojson",weighted)
    summary = {
        "version":"0.24",
        "baseline_v022_fragments":len(baseline),
        "weighted_v024_fragments":len(weighted),
        "baseline_length_px":round(sum(r["line"].length for r in baseline),1),
        "weighted_length_px":round(sum(r["line"].length for r in weighted),1),
        "higher_evidence_fragments":sum(
            r["evidence_tier"]=="higher_evidence" for r in weighted),
        "tower_anchors_shared":len(anchor_info),
        "rejected_tower_components":len(excluded),
        "building_candidates_v05":roof_count,
        "weighted_trace_details":details,
        "warning":"This is an unreviewed experimental candidate layer. "
                  "Thickness and blackness can also describe roads. "
                  "No towers were linked by inference and no wall loop is assumed.",
    }
    (folder/"stats_v024.json").write_text(json.dumps(summary,indent=2),
                                          encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print("Saved:",folder.resolve())


if __name__=="__main__":
    main()
