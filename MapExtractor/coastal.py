"""CityVisualizer v0.9: land/water boundary and coastal layers.

Candidate data only. Reuse the v0.7 mask, refine it conservatively with
local color evidence, then derive shoreline, nearshore and land-side bands.
Image coordinates are pixels; no depth, elevation or confirmed beach type.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, mapping

from extract import read_image, save_png
from water import spatial_mask, find_water_candidates
from shoreline import v07_mask


def hsv_ranges(image, ranges):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    result = np.zeros(image.shape[:2], np.uint8)
    for entry in ranges:
        lower = np.array(entry["lower"], np.uint8)
        upper = np.array(entry["upper"], np.uint8)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("HSV bounds must have three values")
        result |= cv2.inRange(hsv, lower, upper)
    return result


def shore_growth(image, baseline, permitted, settings):
    """Test nearby pixels for water-like color without swallowing known land.

    Local water color is estimated with weighted Gaussian averages. Growth
    is restricted by distance from the v0.7 water, a relaxed HSV water color,
    and an optional confident-land exclusion.
    """
    reach = int(settings.get("max_distance_px", 10))
    if not 0 <= reach <= 40:
        raise ValueError("max_distance_px must be between 0 and 40")
    if reach == 0:
        return baseline.copy(), np.zeros_like(baseline), np.zeros_like(baseline)
    if "water_hsv_ranges" not in settings:
        raise ValueError("boundary_growth.water_hsv_ranges is required")
    confident_land = hsv_ranges(image, settings.get("land_hsv_ranges", []))
    relaxed_water = hsv_ranges(image, settings["water_hsv_ranges"])
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    source = (baseline > 0).astype(np.float32)
    sigma = float(settings.get("local_color_sigma_px", 22))
    if sigma <= 0:
        raise ValueError("local_color_sigma_px must be positive")
    density = cv2.GaussianBlur(source, (0, 0), sigma)
    colors = []
    for channel in range(3):
        weighted = cv2.GaussianBlur(lab[:, :, channel] * source, (0, 0), sigma)
        colors.append(weighted / np.maximum(density, 0.001))
    reference = np.stack(colors, axis=-1)
    tolerance = settings.get("lab_tolerance", [30, 18, 18])
    if len(tolerance) != 3 or any(float(v) <= 0 for v in tolerance):
        raise ValueError("lab_tolerance needs three positive numbers")
    error = np.max(np.abs(lab - reference) /
                   np.asarray(tolerance, np.float32).reshape(1, 1, 3), axis=2)
    nearby = cv2.dilate(baseline, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (reach * 2 + 1, reach * 2 + 1))) > 0
    feasible = (nearby & (permitted > 0) & (relaxed_water > 0) &
                (confident_land == 0) & (density > 0.015) & (error <= 1))
    # Keep only newly selected regions joined to baseline water.
    combined = (baseline > 0) | feasible
    count, labels = cv2.connectedComponents(combined.astype(np.uint8), 8)
    touching = np.unique(labels[baseline > 0])
    keep = np.zeros(count, bool)
    keep[touching] = True
    keep[0] = False
    grown = np.where(keep[labels], 255, 0).astype(np.uint8)
    grown[permitted == 0] = 0
    additions = np.where((grown > 0) & (baseline == 0), 255, 0).astype(np.uint8)
    return grown, additions, confident_land


def coastline_segments(water, permitted, options):
    """Extract physical candidate shoreline, omitting clipped map edges."""
    edge_distance = float(options.get("open_boundary_distance_px", 3))
    min_length = float(options.get("min_segment_length_px", 6))
    simplify = float(options.get("simplify_px", 0.7))
    if edge_distance < 0 or min_length < 0 or simplify < 0:
        raise ValueError("Invalid coast boundary settings")
    # A map crop edge is not an actual coastline.
    inside_distance = cv2.distanceTransform(
        (permitted > 0).astype(np.uint8), cv2.DIST_L2, 3)
    contours, hierarchy = cv2.findContours(water, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_NONE)
    segments = []
    if hierarchy is None:
        return segments
    for index, contour in enumerate(contours):
        pts = contour.reshape(-1, 2)
        if len(pts) < 2:
            continue
        # Cycle the closed ring so clipped runs can be split correctly.
        valid = np.asarray([
            0 <= x < water.shape[1] and 0 <= y < water.shape[0] and
            inside_distance[y, x] > edge_distance for x, y in pts], bool)
        if valid.all():
            runs = [pts]
        else:
            if not valid.any():
                continue
            first_gap = int(np.flatnonzero(~valid)[0])
            pts = np.roll(pts, -first_gap, axis=0)
            valid = np.roll(valid, -first_gap)
            runs = []
            start = None
            for j, is_valid in enumerate(valid):
                if is_valid and start is None:
                    start = j
                if not is_valid and start is not None:
                    runs.append(pts[start:j])
                    start = None
            if start is not None:
                runs.append(pts[start:])
        is_hole = hierarchy[0][index][3] >= 0
        for run in runs:
            if len(run) < 2:
                continue
            if valid.all():
                run = np.vstack((run, run[0]))
            line = LineString(run.tolist()).simplify(simplify,
                                                      preserve_topology=True)
            if line.length >= min_length and not line.is_empty:
                segments.append((line, "island" if is_hole else "mainland"))
    return segments


def layer_polygons(mask, minimum=2):
    """Convert masks into topology-preserving candidate polygons."""
    objects, _ = find_water_candidates(mask, {
        "min_area_px2": minimum, "simplify_fraction": 0.002})
    return [(area, geom) for _, area, geom, _ in objects]


def feature_collection(features):
    return {"type": "FeatureCollection", "features": features}


def polygon_features(items, layer, interpretation):
    features = []
    for n, (area, geom) in enumerate(items, 1):
        fid = f"{layer[:2].upper()}{n:04d}"
        features.append({"type": "Feature", "id": fid,
                         "properties": {
                             "id": fid, "layer": layer,
                             "classification": layer + "_candidate",
                             "interpretation": interpretation,
                             "review_status": "unreviewed",
                             "coordinate_system": "image_pixels_top_left_origin",
                             "area_px2": round(area, 1), "confidence": None
                         }, "geometry": mapping(geom)})
    return features


def build_layers(image, profile):
    options = profile.get("coastal_layers", {})
    allowed = spatial_mask(image.shape, profile)
    baseline = v07_mask(image, profile)
    water, added, confident_land = shore_growth(
        image, baseline, allowed, options.get("boundary_growth", {}))
    segments = coastline_segments(water, allowed, options)
    shore_mask = np.zeros_like(water)
    shore_features = []
    for n, (line, role) in enumerate(segments, 1):
        points = np.rint(np.asarray(line.coords)).astype(np.int32)
        if len(points) < 2:
            continue
        cv2.polylines(shore_mask, [points], False, 255, 1)
        fid = f"SH{n:04d}"
        shore_features.append({"type": "Feature", "id": fid,
            "properties": {"id": fid, "layer": "shoreline",
                           "classification": "shoreline_candidate",
                           "boundary_role": role, "review_status": "unreviewed",
                           "coordinate_system": "image_pixels_top_left_origin",
                           "confidence": None},
            "geometry": mapping(line)})
    # Bands are *cartographic proximity areas*, not a measured bathymetry.
    littoral_width = int(options.get("littoral_band_px", 8))
    coastal_width = int(options.get("coastal_land_band_px", 9))
    if not 0 <= littoral_width <= 100 or not 0 <= coastal_width <= 100:
        raise ValueError("Band widths must be 0..100 pixels")
    def expanded(width):
        if not width:
            return np.zeros_like(shore_mask)
        return cv2.dilate(shore_mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (width*2+1, width*2+1)))
    littoral = cv2.bitwise_and(expanded(littoral_width), water)
    coastal = cv2.bitwise_and(expanded(coastal_width), cv2.bitwise_not(water))
    coastal = cv2.bitwise_and(coastal, allowed)
    coastal_features = polygon_features(layer_polygons(coastal), "coastal_terrain",
                                        "landward_shore_proximity_not_material")
    for feature in coastal_features:
        feature["properties"]["material"] = "unknown"
        feature["properties"]["possible_materials"] = [
            "sand", "gravel", "stone", "mixed", "mud", "vegetation", "quay"]
    return {
        "masks": {"baseline_v07": baseline, "water": water, "added_water": added,
                  "shoreline": shore_mask, "littoral": littoral, "coastal": coastal,
                  "land_evidence": confident_land},
        "layers": {
            "water": polygon_features(layer_polygons(water), "water", "water_surface"),
            "shoreline": shore_features,
            "littoral_zone": polygon_features(layer_polygons(littoral),
                "littoral_zone", "water_side_band_not_proven_shallow"),
            "coastal_terrain": coastal_features
        },
        "line_count": len(shore_features)
    }


def main():
    parser = argparse.ArgumentParser(description="v0.9 water/shore/littoral/coastal candidate export")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/coastal_v09"))
    args = parser.parse_args()
    img = read_image(args.image)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    result = build_layers(img, profile)
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    for name, mask in result["masks"].items():
        save_png(folder / f"{name}_v09.png", mask)
    for name, features in result["layers"].items():
        (folder / f"{name}_v09.geojson").write_text(
            json.dumps(feature_collection(features), indent=2), encoding="utf-8")
    overlay = img.copy()
    tint = img.copy()
    masks = result["masks"]
    tint[masks["water"] > 0] = (195, 125, 55)
    tint[masks["littoral"] > 0] = (205, 220, 70)
    tint[masks["coastal"] > 0] = (35, 170, 220)
    overlay = cv2.addWeighted(img, 0.6, tint, 0.4, 0)
    overlay[masks["shoreline"] > 0] = (0, 0, 245)
    save_png(folder / "coastal_overlay_v09.png", overlay)
    stats = {"version": "0.9", "profile": args.profile.name,
             "v07_water_pixels": int(np.count_nonzero(masks["baseline_v07"])),
             "water_pixels": int(np.count_nonzero(masks["water"])),
             "new_water_pixels": int(np.count_nonzero(masks["added_water"])),
             "feature_counts": {k: len(v) for k, v in result["layers"].items()},
             "warnings": [
                 "Candidate polygons and coastlines require manual review",
                 "Bands are spatial proximity, not known seabed or beach type",
                 "Map boundaries are removed from shoreline candidates",
                 "Any missing v0.7 water may remain missing"
             ]}
    (folder / "coastal_stats_v09.json").write_text(json.dumps(stats, indent=2),
                                                      encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved to: {folder.resolve()}")


if __name__ == "__main__":
    main()
