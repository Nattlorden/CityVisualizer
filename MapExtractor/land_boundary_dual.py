"""v0.12: dual-image land/strand boundary proposals, keeping separate landmasses.

Unlike v0.10-0.11, ordered traces follow contours of land-color regions.
No water geometry is replaced; v0.7 is only a proximity reference.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, mapping

from extract import read_image, save_png
from water import spatial_mask
from shoreline import v07_mask


def color_mask(image, ranges):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], np.uint8)
    for r in ranges:
        lo, hi = np.asarray(r["lower"], np.uint8), np.asarray(r["upper"], np.uint8)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("HSV ranges must each have three channels")
        mask |= cv2.inRange(hsv, lo, hi)
    return mask


def ink_mask(image, options):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    window = int(options.get("ink_window", 19))
    if window < 3 or window % 2 != 1:
        raise ValueError("ink_window must be odd and >=3")
    average = cv2.GaussianBlur(gray, (window, window), 0)
    contrast = int(options.get("ink_contrast", 13))
    dark = (gray.astype(np.int16) < average.astype(np.int16) - contrast)
    return ((dark & (gray < int(options.get("ink_max", 155)))) * 255).astype(np.uint8)


def kernel(radius):
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))


def spatial_distance(allowed):
    return cv2.distanceTransform(np.pad((allowed > 0).astype(np.uint8), 1),
                                 cv2.DIST_L2, 3)[1:-1, 1:-1]


def trace_regions(green, sand, ink, water, allowed, cfg):
    # Smooth only the *land-color evidence*, not the ink geometry.
    smooth = int(cfg.get("green_smooth_px", 3))
    if smooth:
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel(smooth))
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel(max(1, smooth // 2)))
    green &= allowed
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (green > 0).astype(np.uint8), connectivity=8)
    preserved = np.zeros_like(green)
    minimum = int(cfg.get("min_land_area_px2", 175))
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] >= minimum:
            preserved[labels == idx] = 255
    contours, hierarchy = cv2.findContours(
        preserved, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return [], preserved
    sand_near = cv2.dilate(sand, kernel(cfg.get("sand_near_px", 10))) > 0
    water_near = cv2.dilate(water, kernel(cfg.get("water_near_px", 75))) > 0
    ink_near = cv2.dilate(ink, kernel(cfg.get("ink_near_px", 3))) > 0
    safe = spatial_distance(allowed) > float(cfg.get("edge_clearance_px", 3))
    traces = []
    min_run = int(cfg.get("min_run_points", 10))
    for index, contour in enumerate(contours):
        points = contour.reshape(-1, 2)
        if len(points) < min_run:
            continue
        x, y = points[:, 0], points[:, 1]
        nearby = sand_near[y, x] & water_near[y, x] & safe[y, x]
        # Preserve an ordered, contiguous run of points. Do not create a line
        # around each fragment of a skeleton (the v0.11 problem).
        if not nearby.any():
            continue
        if nearby.all():
            runs = [np.vstack((points, points[0]))]
        else:
            first_gap = int(np.flatnonzero(~nearby)[0])
            points = np.roll(points, -first_gap, axis=0)
            nearby = np.roll(nearby, -first_gap)
            runs = []
            start = None
            for j, passed in enumerate(nearby):
                if passed and start is None:
                    start = j
                if not passed and start is not None:
                    runs.append(points[start:j])
                    start = None
            if start is not None:
                runs.append(points[start:])
        for run in runs:
            if len(run) < min_run:
                continue
            sample = run[::max(1, len(run)//150)]
            support = float(np.mean(ink_near[sample[:, 1], sample[:, 0]]))
            line = LineString(run.tolist()).simplify(
                float(cfg.get("simplify_px", 0.75)), preserve_topology=True)
            if line.length < float(cfg.get("min_line_length_px", 15)):
                continue
            traces.append({
                "geometry": line, "ink_support": support,
                "ring_type": "hole" if hierarchy[0][index][3] >= 0 else "exterior",
                "source_component": int(index),
                "closed": bool(nearby.all()),
            })
    return traces, preserved


def analyze(original, hdr, profile):
    if original.shape != hdr.shape:
        raise ValueError("Original and HDR images differ in size; align them first")
    cfg = profile.get("land_boundary_v012")
    if not cfg:
        raise ValueError("Missing land_boundary_v012 profile")
    allowed = spatial_mask(original.shape, profile)
    water = v07_mask(original, profile)
    green_a = color_mask(original, cfg["green_hsv_ranges"])
    green_b = color_mask(hdr, cfg.get("hdr_green_hsv_ranges", cfg["green_hsv_ranges"]))
    sand_a = color_mask(original, cfg["sand_hsv_ranges"])
    sand_b = color_mask(hdr, cfg.get("hdr_sand_hsv_ranges", cfg["sand_hsv_ranges"]))
    ink_a = ink_mask(original, cfg)
    ink_b = ink_mask(hdr, cfg)
    # HSV consensus can change after HDR grading. Accept either image's
    # land/shore evidence but report how much is common to both.
    green = (green_a | green_b) & allowed
    sand = (sand_a | sand_b) & allowed
    ink = ink_a | ink_b
    traces, mass = trace_regions(green, sand, ink, water, allowed, cfg)
    return traces, {
        "green_original": green_a, "green_hdr": green_b,
        "green_consensus": green_a & green_b, "sand_evidence": sand,
        "ink_original": ink_a, "ink_hdr": ink_b,
        "land_regions": mass, "water_v07_reference": water,
    }


def main():
    p = argparse.ArgumentParser(description="v0.12 dual-image land boundary test")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--hdr", required=True, type=Path)
    p.add_argument("--profile", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("output/land_boundary_v012"))
    args = p.parse_args()
    original = read_image(args.image)
    hdr = read_image(args.hdr)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    traces, masks = analyze(original, hdr, profile)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        save_png(args.output / f"{name}.png", mask)
    features = []
    for i, t in enumerate(traces, 1):
        fid = f"LB{i:04d}"
        features.append({"type": "Feature", "id": fid,
            "properties": {
                "id": fid, "classification": "land_boundary_candidate",
                "review_status": "unreviewed", "source_component": t["source_component"],
                "ring_type": t["ring_type"], "closed": t["closed"],
                "ink_support_fraction": round(t["ink_support"], 3),
                "coordinate_system": "image_pixels_top_left_origin",
            }, "geometry": mapping(t["geometry"])})
    for image, name in ((original, "original_overlay"), (hdr, "hdr_overlay")):
        overlay = image.copy()
        for t in traces:
            coords = np.rint(np.asarray(t["geometry"].coords)).astype(np.int32)
            if len(coords) > 1:
                color = (20, 205, 245) if t["ink_support"] >= .35 else (0, 0, 230)
                cv2.polylines(overlay, [coords], False, color, 2)
        save_png(args.output / f"{name}.png", overlay)
    (args.output / "land_boundary_v012.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8")
    report = {"version": "0.12", "experimental": True,
              "candidate_runs": len(features),
              "source_components": len(set(t["source_component"] for t in traces)),
              "closed_runs": sum(t["closed"] for t in traces),
              "supported_by_ink": sum(t["ink_support"] >= .35 for t in traces),
              "warning": "Color-region contours, not guaranteed original ink lines. "
                         "Islands are separate components; manual review required."}
    (args.output / "stats_v012.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
