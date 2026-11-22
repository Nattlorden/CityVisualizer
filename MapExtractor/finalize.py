"""CityVisualizer v0.5: export LAB-normalized building candidates.

The map photograph is unchanged; coordinates are image pixels.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon, mapping
from extract import read_image, save_png, make_mask, failed_filters
from color_compare import local_lab_normalize


def exclude_frame(mask, config):
    result = mask.copy()
    for polygon in config.get("v05_exclusion_polygons", []):
        points = np.asarray(polygon, dtype=np.int32)
        if len(points) < 3:
            raise ValueError("Exclusion polygon requires >= 3 points")
        cv2.fillPoly(result, [points], 0)
    return result


def build_candidates(mask, rules, simplify_fraction):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results, failures = [], {}
    for contour in contours:
        area = float(cv2.contourArea(contour))
        bbox = cv2.boundingRect(contour)
        reasons = failed_filters(area, bbox, rules)
        if reasons:
            for reason in reasons:
                failures[reason] = failures.get(reason, 0) + 1
            continue
        epsilon = max(1.1, cv2.arcLength(contour, True) * simplify_fraction)
        simple = cv2.approxPolyDP(contour, epsilon, True)
        points = [(int(p[0][0]), int(p[0][1])) for p in simple]
        if len(points) < 3:
            continue
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        results.append({"contour": contour, "polygon": poly, "bbox": bbox, "area": area})
    results.sort(key=lambda obj: (obj["bbox"][1] // 25, obj["bbox"][0]))
    return results, len(contours), failures


def main():
    parser = argparse.ArgumentParser(description="v0.5 building candidate export")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("MapExtractor/sample-config.json"))
    parser.add_argument("--output", type=Path, default=Path("output/buildings_v05"))
    args = parser.parse_args()
    image = read_image(args.image)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    corrected = local_lab_normalize(image, config.get("v04_color_normalization", {}))
    normalized_mask = make_mask(corrected, config)
    final_mask = exclude_frame(normalized_mask, config)
    objects, count, failures = build_candidates(
        final_mask, config["v02_filters"], float(config.get("simplify_fraction", .009)))
    args.output.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    features = []
    for index, obj in enumerate(objects, 1):
        identifier = f"C{index:04d}"
        cv2.drawContours(overlay, [obj["contour"]], -1, (0, 200, 255), 2)
        features.append({"type": "Feature", "id": identifier, "properties": {
            "id": identifier, "classification": "building_candidate",
            "review_status": "unreviewed", "source": "lab_normalized_v05",
            "area_px2": round(obj["area"], 1),
            "coordinate_system": "image_pixels_top_left_origin",
            "confidence": None
        }, "geometry": mapping(obj["polygon"])})
    save_png(args.output / "building_mask_v05.png", final_mask)
    save_png(args.output / "building_candidates_v05.png", overlay)
    (args.output / "building_candidates_v05.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "name": "building_candidates_v05",
                    "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {"version": "0.5", "status": "unreviewed_candidates",
             "image_pixels": [image.shape[1], image.shape[0]],
             "mask_pixels_before_exclusions": int(np.count_nonzero(normalized_mask)),
             "mask_pixels_after_exclusions": int(np.count_nonzero(final_mask)),
             "contours_after_exclusions": count,
             "exported_candidates": len(objects), "rejection_reasons": failures,
             "note": "Counts are not verified buildings; IDs are provisional."}
    (args.output / "building_stats_v05.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
