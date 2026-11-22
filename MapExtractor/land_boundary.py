"""MapExtractor v0.11: candidate INNER land/shore ink boundary.

The outer shore/water boundary remains v0.7. This script never turns the
inner line into water; it only exports unreviewed, possibly broken linework.
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


def color_evidence(image, ranges):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    out = np.zeros(image.shape[:2], np.uint8)
    for entry in ranges:
        low = np.asarray(entry["lower"], np.uint8)
        high = np.asarray(entry["upper"], np.uint8)
        if low.shape != (3,) or high.shape != (3,):
            raise ValueError("HSV thresholds must be three-element vectors")
        out |= cv2.inRange(hsv, low, high)
    return out


def disk(radius):
    radius = max(1, int(radius))
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def skeletonize(binary, rounds=24):
    """Small OpenCV-only morphological skeleton."""
    remaining = binary.copy()
    skeleton = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(rounds):
        eroded = cv2.erode(remaining, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton |= cv2.subtract(remaining, opened)
        remaining = eroded
        if not cv2.countNonZero(remaining):
            break
    return skeleton | remaining


def extract_ink(image, water, allowed, opts):
    """Find dark linework close to green-to-sand transitions, NOT water edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    window = int(opts.get("local_window_px", 25))
    if window < 3 or window % 2 != 1:
        raise ValueError("local_window_px must be odd and >=3")
    local = cv2.GaussianBlur(gray, (window, window), 0)
    ink = ((gray.astype(np.int16) <= local.astype(np.int16) -
            int(opts.get("min_contrast", 11))) &
           (gray < int(opts.get("max_gray", 160)))).astype(np.uint8) * 255
    green = color_evidence(image, opts["green_land_hsv_ranges"])
    sand = color_evidence(image, opts["shore_sand_hsv_ranges"])
    # Ignore small scattered green flecks (trees, print texture) when
    # deciding where the main green LAND region ends.
    major = cv2.morphologyEx(green, cv2.MORPH_CLOSE,
                              disk(opts.get("green_close_px", 6)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (major > 0).astype(np.uint8), connectivity=8)
    green_mass = np.zeros_like(green)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= int(opts.get("min_land_region_px", 1100)):
            green_mass[labels == i] = 255
    land_edge = cv2.morphologyEx(
        green_mass, cv2.MORPH_GRADIENT, disk(2))
    adjacency = ((cv2.dilate(land_edge,
                   disk(opts.get("land_proximity_px", 8))) > 0) &
                 (cv2.dilate(sand,
                   disk(opts.get("sand_proximity_px", 11))) > 0))
    outer = cv2.morphologyEx(
        water, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    corridor = cv2.dilate(outer,
                  disk(opts.get("search_width_px", 65))) > 0
    safe = cv2.distanceTransform(
        np.pad((allowed > 0).astype(np.uint8), 1),
        cv2.DIST_L2, 3)[1:-1, 1:-1]
    possible = ((ink > 0) & adjacency & corridor &
                (safe > float(opts.get("map_edge_clearance_px", 4))))
    candidate = possible.astype(np.uint8) * 255
    gap = int(opts.get("gap_close_px", 1))
    if gap > 0:
        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_CLOSE, disk(gap))
    return {
        "dark_ink": ink, "major_green_land": green_mass,
        "sand_color": sand, "land_edge_evidence": land_edge,
        "inner_boundary_candidates": skeletonize(candidate),
    }


def candidate_lines(mask, opts):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    result = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < int(opts.get("min_component_pixels", 7)):
            continue
        component = ((labels == i).astype(np.uint8) * 255)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            points = contour.reshape(-1, 2)
            if len(points) < 3:
                continue
            # A provisional trace around the skeleton component; this is not
            # a smooth, ordered shoreline. It must be reviewed or retraced.
            line = LineString(points.tolist()).simplify(
                float(opts.get("simplify_px", 0.65)))
            if line.length >= float(opts.get("min_line_length_px", 6)):
                result.append(line)
    return result


def main():
    parser = argparse.ArgumentParser(description="v0.11: inner land/shore line candidates")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--water-mask", type=Path,
                        help="Optional saved v0.7 mask, same image dimensions")
    parser.add_argument("--output", type=Path,
                        default=Path("output/land_boundary_v011"))
    args = parser.parse_args()
    image = read_image(args.image)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    allowed = spatial_mask(image.shape, profile)
    water = v07_mask(image, profile) if args.water_mask is None else         cv2.cvtColor(read_image(args.water_mask), cv2.COLOR_BGR2GRAY)
    if water.shape != image.shape[:2]:
        raise ValueError("Water mask dimensions differ from the source image")
    water = np.where(water > 127, 255, 0).astype(np.uint8)
    opts = profile.get("land_boundary_v011")
    if not opts:
        raise ValueError("Missing land_boundary_v011 profile section")
    masks = extract_ink(image, water, allowed, opts)
    lines = candidate_lines(masks["inner_boundary_candidates"], opts)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        save_png(args.output / (name + ".png"), mask)
    outer = cv2.morphologyEx(water, cv2.MORPH_GRADIENT,
                             np.ones((3, 3), np.uint8))
    overlay = image.copy()
    overlay[outer > 0] = (245, 130, 20)  # BLUE: v0.7 outer water edge
    features = []
    for index, line in enumerate(lines, 1):
        points = np.rint(np.asarray(line.coords)).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(overlay, [points], False, (0, 0, 245), 2)
        ident = f"LB{index:04d}"
        features.append({
            "type": "Feature", "id": ident,
            "properties": {
                "id": ident, "classification": "land_boundary_candidate",
                "review_status": "unreviewed",
                "source_method": "dark_ink_near_green_sand_v011",
                "coordinate_system": "image_pixels_top_left_origin",
            }, "geometry": mapping(line),
        })
    save_png(args.output / "inner_vs_outer_v011.png", overlay)
    (args.output / "land_boundary_v011.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features},
                   indent=2), encoding="utf-8")
    stats = {
        "version": "0.11",
        "candidate_lines": len(lines),
        "candidate_pixels": int(np.count_nonzero(masks["inner_boundary_candidates"])),
        "status": "unreviewed_fragments",
        "warning": "Do not infer a closed coastal zone or coastline from this draft.",
    }
    (args.output / "stats_v011.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
