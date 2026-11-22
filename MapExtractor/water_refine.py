"""CityVisualizer v0.7: refine color-map water candidates without hardcoding map data.

Produces a merged candidate mask, a transparent QA overlay, and GeoJSON.
All pixels remain in image coordinates. This is not a coastline guarantee.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import mapping

from extract import read_image, save_png
from water import make_water_mask, spatial_mask, find_water_candidates


def hsv_mask(image, ranges):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    result = np.zeros(image.shape[:2], np.uint8)
    for item in ranges:
        lower, upper = np.asarray(item["lower"], np.uint8), np.asarray(item["upper"], np.uint8)
        result |= cv2.inRange(hsv, lower, upper)
    return result


def grow_near_water(image, base, profile, permitted):
    settings = profile.get("shore_refinement", {})
    pixels = int(settings.get("distance_px", 0))
    if pixels <= 0:
        return np.zeros_like(base)
    if pixels > 50:
        raise ValueError("distance_px > 50 is not supported")
    radius = pixels * 2 + 1
    neighborhood = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)))
    relaxed = hsv_mask(image, settings["hsv_ranges"])
    extra = cv2.bitwise_and(relaxed, neighborhood)
    extra = cv2.bitwise_and(extra, permitted)
    # Do not convert every dark edge into water: keep only components
    # connected to the original candidate mask.
    merged = cv2.bitwise_or(base, extra)
    count, labels = cv2.connectedComponents((merged > 0).astype(np.uint8), connectivity=8)
    touches = np.unique(labels[base > 0])
    keep = np.zeros(count, dtype=bool)
    keep[touches] = True
    keep[0] = False
    return np.where(keep[labels] & (base == 0), 255, 0).astype(np.uint8)


def seed_lakes(image, profile, permitted):
    settings = profile.get("seeded_water", {})
    points = settings.get("points", [])
    result = np.zeros(image.shape[:2], np.uint8)
    if not points:
        return result, []
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    radius = int(settings.get("max_radius_px", 30))
    tolerance = settings.get("lab_tolerance", [9, 9, 9])
    if radius < 1 or radius > 200 or len(tolerance) != 3:
        raise ValueError("Invalid seed settings")
    blur = cv2.GaussianBlur(lab, (3, 3), 0)
    h, w = result.shape
    report = []
    for idx, pair in enumerate(points):
        x, y = map(int, pair)
        if not (0 <= x < w and 0 <= y < h) or permitted[y, x] == 0:
            report.append({"point": [x, y], "status": "outside_region"})
            continue
        local_limit = np.zeros((h, w), np.uint8)
        cv2.circle(local_limit, (x, y), radius, 255, -1)
        local_limit = cv2.bitwise_and(local_limit, permitted)
        # OpenCV floodFill mask: zero = permissible, nonzero = blocked.
        block = np.pad(np.where(local_limit > 0, 0, 1).astype(np.uint8), 1, constant_values=1)
        block *= 1
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
        lo = tuple(map(int, tolerance))
        count, _, out_mask, _ = cv2.floodFill(blur, block, (x, y), (0, 0, 0), lo, lo, flags)
        region = (out_mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
        # Optional tiny stray artifacts remain separately inspectable.
        result = cv2.bitwise_or(result, region)
        report.append({"point": [x, y], "pixels": int(count), "status": "processed"})
    return result, report


def main():
    p = argparse.ArgumentParser(description="v0.7 water candidate refinement")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("output/water_v07"))
    args = p.parse_args()
    image = read_image(args.image)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    base = make_water_mask(image, profile)
    permitted = spatial_mask(image.shape, profile)
    shore = grow_near_water(image, base, profile, permitted)
    seeds, seed_report = seed_lakes(image, profile, permitted)
    result = cv2.bitwise_or(cv2.bitwise_or(base, shore), seeds)
    result = cv2.bitwise_and(result, permitted)
    candidates, contour_total = find_water_candidates(result, profile)
    overlay = image.copy()
    # BGR: existing water blue, shoreline additions green, seeded areas orange.
    tint = image.copy()
    tint[base > 0] = (230, 100, 15)
    tint[shore > 0] = (40, 220, 50)
    tint[seeds > 0] = (15, 150, 245)
    overlay = cv2.addWeighted(image, 0.65, tint, 0.35, 0)
    features = []
    for index, (bbox, area, geom, contour) in enumerate(candidates, 1):
        ident = f"W{index:04d}"
        cv2.drawContours(overlay, [contour], -1, (255, 45, 20), 1)
        features.append({"type": "Feature", "id": ident,
                         "properties": {"id": ident, "classification": "water_candidate",
                                        "review_status": "unreviewed",
                                        "source_method": "hsv_shore_and_seeds_v07",
                                        "area_px2": round(area, 1),
                                        "coordinate_system": "image_pixels_top_left_origin",
                                        "confidence": None},
                         "geometry": mapping(geom)})
    args.output.mkdir(parents=True, exist_ok=True)
    save_png(args.output / "water_mask_v07.png", result)
    save_png(args.output / "water_overlay_v07.png", overlay)
    save_png(args.output / "water_new_shore_v07.png", shore)
    save_png(args.output / "water_seeded_v07.png", seeds)
    (args.output / "water_candidates_v07.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")
    stats = {"version": "0.7", "profile": args.profile.name,
             "baseline_water_pixels": int(np.count_nonzero(base)),
             "shore_pixels_added": int(np.count_nonzero(shore & ~base)),
             "seeded_pixels": int(np.count_nonzero(seeds)),
             "final_water_pixels": int(np.count_nonzero(result)),
             "contours": contour_total, "water_candidates": len(features),
             "seed_report": seed_report,
             "note": "All candidates unreviewed; water islands and channels require visual validation."}
    (args.output / "water_stats_v07.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
