"""CityVisualizer v0.6: configurable candidate water-layer extraction.

Default is HSV color segmentation; monochrome maps are explicitly unsupported
by this method and should use an alternate profile/method in a later version.
Image pixel coordinates (top-left origin), not geographic coordinates.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, mapping

from extract import read_image, save_png


def valid_polygon(coords):
    points = np.asarray(coords, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Expected polygon containing at least three [x,y] points")
    return points


def spatial_mask(shape, profile):
    height, width = shape[:2]
    result = np.full((height, width), 255, np.uint8)
    if profile.get("include_polygons"):
        result[:] = 0
        for coords in profile["include_polygons"]:
            cv2.fillPoly(result, [valid_polygon(coords)], 255)
    for coords in profile.get("exclude_polygons", []):
        cv2.fillPoly(result, [valid_polygon(coords)], 0)
    return result


def make_water_mask(image, profile):
    method = profile.get("method", "hsv_ranges")
    if method != "hsv_ranges":
        raise ValueError(f"Unsupported method {method!r}; v0.6 supports hsv_ranges only")
    ranges = profile.get("hsv_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("hsv_ranges must be a nonempty list")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], np.uint8)
    for entry in ranges:
        lo, hi = np.array(entry["lower"], np.int32), np.array(entry["upper"], np.int32)
        if lo.shape != (3,) or hi.shape != (3,) or np.any(lo > hi):
            raise ValueError("Invalid HSV bounds")
        if np.any(lo < 0) or np.any(hi > [179, 255, 255]):
            raise ValueError("HSV values out of OpenCV ranges")
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo.astype(np.uint8), hi.astype(np.uint8)))
    blur = int(profile.get("median_kernel", 0))
    if blur > 1:
        if blur % 2 != 1:
            raise ValueError("median_kernel must be odd")
        mask = cv2.medianBlur(mask, blur)
    close = int(profile.get("closing_kernel", 0))
    if close > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.bitwise_and(mask, spatial_mask(image.shape, profile))
    return mask


def to_geometry(contour, children, simplify_fraction):
    epsilon = max(0.5, cv2.arcLength(contour, True) * simplify_fraction)
    shell = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).tolist()
    if len(shell) < 3:
        return None
    holes = []
    for child in children:
        pts = cv2.approxPolyDP(child, max(0.5, cv2.arcLength(child, True) * simplify_fraction), True).reshape(-1, 2).tolist()
        if len(pts) >= 3:
            holes.append(pts)
    poly = Polygon(shell, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return None if poly.is_empty else poly


def find_water_candidates(mask, profile):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], 0
    parentage = hierarchy[0]
    min_area = float(profile.get("min_area_px2", 80))
    simplify = float(profile.get("simplify_fraction", 0.003))
    candidates = []
    for idx, contour in enumerate(contours):
        if parentage[idx][3] >= 0:
            continue
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        holes = [contours[j] for j in range(len(contours)) if parentage[j][3] == idx]
        geom = to_geometry(contour, holes, simplify)
        if geom is None:
            continue
        bbox = cv2.boundingRect(contour)
        candidates.append((bbox, area, geom, contour))
    candidates.sort(key=lambda it: (it[0][1], it[0][0]))
    return candidates, len(contours)


def main():
    parser = argparse.ArgumentParser(description="Extract a provisional water layer from a color map")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True, help="Map-specific settings, not application code")
    parser.add_argument("--output", type=Path, default=Path("output/water_v06"))
    args = parser.parse_args()
    image = read_image(args.image)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    mask = make_water_mask(image, profile)
    candidates, contours = find_water_candidates(mask, profile)
    args.output.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    features = []
    for index, (bbox, area, geom, contour) in enumerate(candidates, 1):
        identifier = f"W{index:04d}"
        cv2.drawContours(overlay, [contour], -1, (225, 35, 25), 2)
        features.append({"type": "Feature", "id": identifier, "properties": {
            "id": identifier, "classification": "water_candidate",
            "review_status": "unreviewed", "source_method": "hsv_ranges",
            "source_profile": args.profile.name,
            "area_px2": round(area, 1),
            "coordinate_system": "image_pixels_top_left_origin",
            "confidence": None
        }, "geometry": mapping(geom)})
    save_png(args.output / "water_mask_v06.png", mask)
    save_png(args.output / "water_candidates_v06.png", overlay)
    (args.output / "water_candidates_v06.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "name": "water_candidates_v06",
                    "features": features}, indent=2), encoding="utf-8")
    stats = {"version": "0.6", "profile": args.profile.name,
             "method": profile.get("method", "hsv_ranges"),
             "mask_pixels": int(np.count_nonzero(mask)),
             "contours": contours, "candidates": len(features),
             "status": "unreviewed_candidates"}
    (args.output / "water_stats_v06.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
