"""CityVisualizer MapExtractor v0.1 – configurable candidate extraction.

Usage (from repository root):
  .venv\\Scripts\\python.exe MapExtractor\\extract.py --image maps\\original\\harn_sample.jpeg --config MapExtractor\\sample-config.json

Output image coordinates are PIXELS, not real-world coordinates.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, mapping


def read_image(path: Path) -> np.ndarray:
    # imdecode handles non-ASCII filenames on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def analyze(image: np.ndarray, config: dict):
    height, width = image.shape[:2]
    roi = np.zeros((height, width), dtype=np.uint8)
    vertices = config.get("roi_polygon")
    if vertices:
        cv2.fillPoly(roi, [np.asarray(vertices, np.int32)], 255)
    else:
        roi[:] = 255

    # In the sample, cream-colored roofs have high brightness and relatively
    # neutral RGB; this rule is only a starting point, not a universal parser.
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
        selected = cv2.medianBlur(selected, k if k % 2 else k+1)
    close = int(config.get("closing_kernel", 0))
    if close > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = float(config.get("min_area_px", 40))
    max_area = float(config.get("max_area_px", 10000))
    min_fill = float(config.get("min_fill_ratio", 0.32))
    min_side = float(config.get("min_bbox_side_px", 4))
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_side or h < min_side or area / (w*h) < min_fill:
            continue
        epsilon = max(1.1, cv2.arcLength(contour, True) * float(config.get("simplify_fraction", .009)))
        simple = cv2.approxPolyDP(contour, epsilon, True)
        points = [(int(p[0][0]), int(p[0][1])) for p in simple]
        if len(points) < 3:
            continue
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        candidates.append((poly, contour, (x, y, w, h), area))
    # Stable IDs following reading order (top to bottom, then left to right).
    candidates.sort(key=lambda c: (c[2][1]//25, c[2][0]))
    return selected, candidates


def main():
    p = argparse.ArgumentParser(description="Extract likely building footprints from an illustrated map")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path, default=Path("output"))
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    image = read_image(args.image)
    mask, candidates = analyze(image, config)
    args.output.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    features = []
    for i, (poly, contour, bbox, area) in enumerate(candidates, 1):
        id_ = f"B{i:04d}"
        cv2.drawContours(overlay, [contour], -1, (0, 15, 230), 2)
        x, y, w, h = bbox
        if bool(config.get("draw_labels", True)):
            cv2.putText(overlay, str(i), (x, max(12, y-2)), cv2.FONT_HERSHEY_SIMPLEX, .32, (25, 20, 190), 1, cv2.LINE_AA)
        features.append({"type": "Feature", "id": id_, "properties": {
            "id": id_, "classification": "building_candidate", "review_status": "unreviewed",
            "source": "color_threshold_v0.1", "area_px2": round(area, 1),
            "coordinate_system": "image_pixels_top_left_origin",
            "confidence": None
        }, "geometry": mapping(poly)})
    cv2.imencode(".png", mask)[1].tofile(str(args.output / "candidate_mask.png"))
    cv2.imencode(".png", overlay)[1].tofile(str(args.output / "building_candidates.png"))
    collection = {"type": "FeatureCollection", "name": "building_candidates", "features": features}
    (args.output / "building_candidates.geojson").write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Image: {image.shape[1]} x {image.shape[0]} px")
    print(f"Candidate footprints: {len(features)} (NOT verified buildings)")
    print(f"Saved to: {args.output.resolve()}")

if __name__ == "__main__":
    main()
