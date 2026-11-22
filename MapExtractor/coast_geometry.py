"""CityVisualizer v0.17: conservative geometric audit of the v0.16 candidate.

Only two changes: (1) break unsupported long shortcuts/triangle tips;
(2) quarantine suspicious small rounded water-feature traces for review.
Do not alter the v0.7 water mask, baseline v0.16, or detected HDR ink.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, mapping

from coast_refine import refine
from extract import read_image, save_png


def make_ink_neighborhood(ink, radius):
    radius = int(radius)
    if not 0 <= radius <= 8:
        raise ValueError("ink_support_radius_px must be 0..8")
    if radius == 0:
        return ink > 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(ink, kernel) > 0


def segment_ink_fraction(a, b, ink_near):
    """Measure how much of the connecting straight line has actual ink."""
    distance = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
    count = max(2, int(math.ceil(distance * 2)) + 1)
    coords = np.rint(np.linspace(a, b, count)).astype(int)
    h, w = ink_near.shape
    inside = ((coords[:, 0] >= 0) & (coords[:, 0] < w) &
              (coords[:, 1] >= 0) & (coords[:, 1] < h))
    if not np.all(inside):
        return 0.0
    return float(np.mean(ink_near[coords[:, 1], coords[:, 0]]))


def audit_polyline(line, ink_near, settings):
    """Only remove an acute unsupported tip, then split unsupported shortcuts.

    Keep all rejected pieces in the review list, rather than deleting them.
    Preserve sharp supported quay corners.
    """
    coords = np.asarray(line.coords, dtype=float)
    if len(coords) < 2:
        return [], [], 0
    minimum_spur_height = float(settings.get("min_spur_height_px", 8))
    maximum_spur_base = float(settings.get("max_spur_base_px", 9))
    ratio = float(settings.get("min_spur_detour_ratio", 2.9))
    spur_support = float(settings.get("max_spur_ink_support", 0.2))
    removed_tips = []
    cleaned = [coords[0]]
    for index in range(1, len(coords)-1):
        previous = cleaned[-1]
        tip = coords[index]
        following = coords[index+1]
        base = np.linalg.norm(following-previous)
        height = np.linalg.norm(tip-previous) if base < 1e-6 else abs(
            np.cross(following-previous, tip-previous)) / base
        detour = (np.linalg.norm(tip-previous) +
                  np.linalg.norm(following-tip))
        support = min(segment_ink_fraction(previous, tip, ink_near),
                      segment_ink_fraction(tip, following, ink_near))
        if (base <= maximum_spur_base and
                height >= minimum_spur_height and
                detour > ratio * max(base, 0.5) and
                support <= spur_support):
            removed_tips.append((previous.copy(), tip.copy(),
                                 following.copy()))
            continue
        cleaned.append(tip)
    cleaned.append(coords[-1])
    max_shortcut = float(settings.get("max_unsupported_link_px", 15))
    min_coverage = float(settings.get("min_link_ink_fraction", 0.28))
    kept = []
    unsupported = []
    current = [cleaned[0]]
    for point in cleaned[1:]:
        previous = current[-1]
        length = float(np.linalg.norm(point-previous))
        coverage = segment_ink_fraction(previous, point, ink_near)
        if length > max_shortcut and coverage < min_coverage:
            unsupported.append((previous.copy(), point.copy(), coverage))
            if len(current) >= 2:
                kept.append(LineString(current))
            current = [point]
        else:
            if not np.array_equal(current[-1], point):
                current.append(point)
    if len(current) >= 2:
        kept.append(LineString(current))
    return kept, unsupported, removed_tips


def round_water_features(water, settings):
    """Diagnostic for small round blobs in v0.7, never delete the water."""
    minimum = float(settings.get("round_min_area_px2", 25))
    maximum = float(settings.get("round_max_area_px2", 2500))
    compactness_min = float(settings.get("round_min_compactness", 0.30))
    maximum_aspect = float(settings.get("round_max_aspect_ratio", 2.4))
    if not 0 <= minimum <= maximum or maximum_aspect < 1:
        raise ValueError("Invalid small round water parameters")
    contours, hierarchy = cv2.findContours(
        water, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(water)
    findings = []
    if hierarchy is None:
        return mask, findings
    for index, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        if not minimum <= area <= maximum:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)
        if perimeter < 1 or w < 3 or h < 3:
            continue
        compactness = 4 * math.pi * area / (perimeter * perimeter)
        aspect = max(w, h) / max(min(w, h), 1)
        if compactness < compactness_min or aspect > maximum_aspect:
            continue
        cv2.drawContours(mask, [contour], -1, 255, 1)
        findings.append({
            "contour_id": index,
            "bbox": [int(x), int(y), int(w), int(h)],
            "area_px2": round(area, 1),
            "compactness": round(compactness, 3),
            "hole_of_water": bool(hierarchy[0][index][3] >= 0),
        })
    return mask, findings


def near_small_round_feature(line, feature_mask, settings):
    radius = int(settings.get("round_review_radius_px", 24))
    if not 0 <= radius <= 80:
        raise ValueError("round_review_radius_px must be 0..80")
    if not np.any(feature_mask):
        return False
    near = cv2.dilate(feature_mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))) > 0 if radius else feature_mask > 0
    coords = np.asarray(line.coords, dtype=float)
    count = max(2, int(math.ceil(line.length / 2)) + 1)
    if len(coords) < 2:
        return False
    distances = np.linalg.norm(np.diff(coords,axis=0),axis=1)
    targets = np.linspace(0, distances.sum(), count)
    cumulative = np.r_[0, np.cumsum(distances)]
    xs = np.interp(targets, cumulative, coords[:, 0])
    ys = np.interp(targets, cumulative, coords[:, 1])
    xi, yi = np.rint(xs).astype(int), np.rint(ys).astype(int)
    h, w = near.shape
    good = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    fraction = np.mean(near[yi[good],xi[good]]) if np.any(good) else 0.0
    return (line.length <= float(settings.get("round_review_max_line_px", 125))
            and fraction >= float(settings.get("round_review_fraction", 0.70)))


def audit(items, ink, water, settings):
    ink_near = make_ink_neighborhood(
        ink, settings.get("ink_support_radius_px", 2))
    round_mask, round_features = round_water_features(water, settings)
    accepted, review, changes = [], [], []
    for item in items:
        pieces, unsupported, tips = audit_polyline(
            item["line"], ink_near, settings)
        for tip in tips:
            changes.append({"source_id": item["source_id"],
                            "reason": "unsupported_triangle_tip",
                            "point": [float(v) for v in tip[1]]})
        for start, stop, support in unsupported:
            changes.append({"source_id": item["source_id"],
                            "reason": "unsupported_long_link",
                            "from": start.tolist(), "to": stop.tolist(),
                            "ink_fraction": round(support, 3)})
        for piece in pieces:
            properties = {key: value for key, value in item.items()
                          if key != "line"}
            properties["parent_source_id"] = item["source_id"]
            properties["line"] = piece
            if near_small_round_feature(piece, round_mask, settings):
                properties["review_reason_v017"] = "near_small_round_water"
                review.append(properties)
            else:
                accepted.append(properties)
    return accepted, review, round_mask, round_features, changes


def feature_collection(items, prefix, category):
    features = []
    for index, item in enumerate(items, 1):
        fid = f"{prefix}{index:04d}"
        properties = {key: value for key, value in item.items()
                      if key != "line"}
        properties.update({"id": fid, "layer": category,
                           "review_status_v017": (
                               "needs_review" if category.endswith("review") else "candidate"),
                           "coordinate_system": "image_pixels_top_left_origin"})
        features.append({"type": "Feature", "id": fid,
                         "properties": properties, "geometry": mapping(item["line"])})
    return {"type": "FeatureCollection", "features": features}


def main():
    parser = argparse.ArgumentParser(description="v0.17 geometry audit, v0.16 preserved")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--hdr", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--water-mask", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("output/coast_geometry_v017"))
    args = parser.parse_args()
    original, hdr = read_image(args.image), read_image(args.hdr)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    settings = profile["coast_geometry_v017"]
    override = None
    if args.water_mask is not None:
        override = cv2.imread(str(args.water_mask), cv2.IMREAD_GRAYSCALE)
        if override is None or override.shape != original.shape[:2]:
            raise ValueError("Water mask missing or image size mismatch")
        override = (override > 127).astype(np.uint8) * 255
    masks, items, _, baseline_stats = refine(
        original, hdr, profile, profile["coast_refine_v016"], override)
    accepted, review, round_mask, features, changes = audit(
        items, masks["ink_hdr_raw"], masks["water_v07"], settings)
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    baseline = hdr.copy()
    result = hdr.copy()
    edge = cv2.morphologyEx(masks["water_v07"], cv2.MORPH_GRADIENT,
                            np.ones((3,3), np.uint8))
    baseline[edge > 0] = (240,130,20)
    result[edge > 0] = (240,130,20)
    for item in items:
        points = np.rint(np.asarray(item["line"].coords)).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(baseline, [points], False, (0,0,230), 1)
    for item in accepted:
        points = np.rint(np.asarray(item["line"].coords)).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(result, [points], False, (0,0,230), 1)
    for item in review:
        points = np.rint(np.asarray(item["line"].coords)).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(result, [points], False, (0,200,240), 2)
    removed_mask = np.zeros_like(round_mask)
    for change in changes:
        if change["reason"] == "unsupported_long_link":
            a = tuple(np.rint(change["from"]).astype(int))
            b = tuple(np.rint(change["to"]).astype(int))
            cv2.line(removed_mask, a, b, 255, 1)
        else:
            p = tuple(np.rint(change["point"]).astype(int))
            cv2.circle(removed_mask, p, 2, 255, -1)
    result[removed_mask>0] = (205,0,205)
    save_png(folder/"hdr_reference_v016.png", baseline)
    save_png(folder/"hdr_geometry_v017.png", result)
    save_png(folder/"suspect_round_water_v017.png", round_mask)
    save_png(folder/"geometry_changes_v017.png", removed_mask)
    save_png(folder/"water_v07_unchanged.png", masks["water_v07"])
    (folder/"coast_candidates_v017.geojson").write_text(
        json.dumps(feature_collection(accepted, "CR", "inner_coast_candidate"),
                   indent=2), encoding="utf-8")
    (folder/"coast_review_v017.geojson").write_text(
        json.dumps(feature_collection(review, "RV", "inner_coast_review"),
                   indent=2), encoding="utf-8")
    (folder/"geometry_changes_v017.json").write_text(
        json.dumps(changes, indent=2), encoding="utf-8")
    stats = {"version":"0.17", "v016_segments": len(items),
             "retained_segments": len(accepted),
             "round_feature_review_segments": len(review),
             "suspect_small_water_contours": len(features),
             "unsupported_links_or_spurs": len(changes),
             "baseline_stats": baseline_stats,
             "notes": ["Does not change the water mask or v0.16.",
                       "Yellow arcs are quarantined for review, not erased.",
                       "Only unsupported shortcuts are split; strong ink-backed quays stay."]}
    (folder/"stats_v017.json").write_text(json.dumps(stats,indent=2),
                                         encoding="utf-8")
    print(json.dumps(stats,indent=2))


if __name__ == "__main__":
    main()
