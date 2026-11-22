"""CityVisualizer MapExtractor v0.2: compare two selections from exactly one mask.

Coordinates are IMAGE PIXELS, origin at top-left. No geographic CRS.
From the project root:
  .venv\\Scripts\\python.exe MapExtractor\\extract.py --image maps\\original\\harn_sample.jpeg --config MapExtractor\\sample-config.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, mapping


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def save_png(path: Path, image: np.ndarray):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise IOError(f"Could not encode {path}")
    encoded.tofile(str(path))


def make_mask(image: np.ndarray, config: dict) -> np.ndarray:
    """Identical threshold + ROI + postprocessing logic to v0.1."""
    height, width = image.shape[:2]
    roi = np.zeros((height, width), dtype=np.uint8)
    vertices = config.get("roi_polygon")
    if vertices:
        cv2.fillPoly(roi, [np.asarray(vertices, np.int32)], 255)
    else:
        roi[:] = 255
    b, g, r = cv2.split(image.astype(np.int16))
    t = config.get("thresholds", {})
    selected = (
        (r >= t.get("min_red", 195)) &
        (g >= t.get("min_green", 185)) &
        (b >= t.get("min_blue", 160)) &
        ((r - g) <= t.get("max_red_green", 42)) &
        ((g - b) <= t.get("max_green_blue", 45)) &
        (roi > 0)
    ).astype(np.uint8) * 255
    k = int(config.get("median_kernel", 3))
    if k > 1:
        selected = cv2.medianBlur(selected, k if k % 2 else k + 1)
    close = int(config.get("closing_kernel", 0))
    if close > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)
    return selected


def failed_filters(area: float, bbox: tuple, rules: dict) -> list[str]:
    x, y, w, h = bbox
    failures = []
    if area < float(rules.get("min_area_px", 35)):
        failures.append("below_min_area")
    if area > float(rules.get("max_area_px", 14000)):
        failures.append("above_max_area")
    if w < float(rules.get("min_bbox_side_px", 4)) or h < float(rules.get("min_bbox_side_px", 4)):
        failures.append("below_min_bbox_side")
    if area / (w * h) < float(rules.get("min_fill_ratio", .32)):
        failures.append("below_min_fill_ratio")
    return failures


def extract(image: np.ndarray, config: dict):
    mask = make_mask(image, config)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    old_rules = config["baseline_filters"]
    new_rules = config["v02_filters"]
    simplify_fraction = float(config.get("simplify_fraction", .009))
    results = []
    rejected = Counter()
    for contour in contours:
        area = float(cv2.contourArea(contour))
        bbox = cv2.boundingRect(contour)
        old_failures = failed_filters(area, bbox, old_rules)
        new_failures = failed_filters(area, bbox, new_rules)
        # Do not alter baseline selection in response to changes to new rules.
        if new_failures and old_failures:
            rejected.update(new_failures)
            continue
        epsilon = max(1.1, cv2.arcLength(contour, True) * simplify_fraction)
        simple = cv2.approxPolyDP(contour, epsilon, True)
        points = [(int(p[0][0]), int(p[0][1])) for p in simple]
        if len(points) < 3:
            rejected["invalid_polygon"] += 1
            continue
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            rejected["invalid_polygon"] += 1
            continue
        status = "baseline" if not old_failures else "new_v02"
        # If one intentionally chooses stricter v02 filters, preserve baseline.
        if new_failures and not old_failures:
            status = "baseline"
        results.append({"polygon": poly, "contour": contour, "bbox": bbox,
                        "area": area, "status": status, "baseline_failures": old_failures})
    # Same ordering as v0.1 for all baseline contours. IDs for new objects
    # are prefixed N to avoid renumbering the previously extracted buildings.
    original = sorted((x for x in results if x["status"] == "baseline"),
                      key=lambda x: (x["bbox"][1] // 25, x["bbox"][0]))
    additions = sorted((x for x in results if x["status"] == "new_v02"),
                       key=lambda x: (x["bbox"][1] // 25, x["bbox"][0]))
    for idx, item in enumerate(original, 1):
        item["id"] = f"B{idx:04d}"
    for idx, item in enumerate(additions, 1):
        item["id"] = f"N{idx:04d}"
    return mask, original, additions, rejected, len(contours)


def main():
    parser = argparse.ArgumentParser(description="MapExtractor v0.2: expanded selection on unchanged map mask")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("MapExtractor/sample-config.json"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    image = read_image(args.image)
    mask, baseline, additions, rejected, contour_count = extract(image, config)
    args.output.mkdir(parents=True, exist_ok=True)

    # Red: v0.1 baseline; yellow: newly accepted in v0.2. Draw new
    # objects last to make even tiny additions visible.
    overlay = image.copy()
    for item in baseline:
        cv2.drawContours(overlay, [item["contour"]], -1, (0, 15, 230), 2)
    for item in additions:
        cv2.drawContours(overlay, [item["contour"]], -1, (0, 235, 255), 1)
    if bool(config.get("draw_labels", True)):
        for item in baseline:
            x, y, w, h = item["bbox"]
            cv2.putText(overlay, item["id"][1:].lstrip('0') or '0',
                        (x, max(12, y - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        .32, (25, 20, 190), 1, cv2.LINE_AA)
        if bool(config.get("draw_new_labels", False)):
            for item in additions:
                x, y, w, h = item["bbox"]
                cv2.putText(overlay, item["id"], (x, max(12, y - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, .25, (0, 100, 180), 1, cv2.LINE_AA)
    # The v0.1 IDs remain identical; new IDs are Nxxxx.
    features = []
    for item in baseline + additions:
        fid = item["id"]
        features.append({"type": "Feature", "id": fid, "properties": {
            "id": fid, "classification": "building_candidate", "review_status": "unreviewed",
            "selection": item["status"], "baseline_failed_filters": item["baseline_failures"],
            "source": "color_threshold_v0.2_same_mask", "area_px2": round(item["area"], 1),
            "coordinate_system": "image_pixels_top_left_origin", "confidence": None,
        }, "geometry": mapping(item["polygon"])})
    save_png(args.output / "candidate_mask.png", mask)
    save_png(args.output / "building_candidates_v02.png", overlay)
    collection = {"type": "FeatureCollection", "name": "building_candidates_v02", "features": features}
    (args.output / "building_candidates_v02.geojson").write_text(
        json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "image_pixels": [image.shape[1], image.shape[0]],
        "mask_unchanged_from_v01_logic": True,
        "contours_in_mask": contour_count,
        "baseline_candidates": len(baseline),
        "new_candidates": len(additions),
        "total_candidates": len(features),
        "new_candidates_failed_v01_filters": dict(Counter(reason for item in additions for reason in item["baseline_failures"])),
        "still_rejected_by_v02_filter": dict(rejected),
        "note": "Filter rejection counts may overlap; a contour can fail multiple filters. Objects are unverified.",
    }
    (args.output / "selection_stats_v02.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Image: {image.shape[1]} x {image.shape[0]} px")
    print(f"Mask contours: {contour_count}")
    print(f"v0.1 baseline: {len(baseline)} | new in v0.2: {len(additions)} | total: {len(features)}")
    print(f"New candidates failed baseline filters: {stats['new_candidates_failed_v01_filters']}")
    print(f"Saved to: {args.output.resolve()}")

if __name__ == "__main__":
    main()
