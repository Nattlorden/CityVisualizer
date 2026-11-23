"""CityVisualizer v0.23: compare building-aware wall extraction against v0.22.

Uses accepted v0.5-style LAB-normalized building candidates (from the ORIGINAL
photo), rather than a new roof color guess. Suppresses their outlines in a
copy of the v0.22 black mask, protects tower anchors and potentially attached
fortification buildings, and compares three tracing variants.

No walls are invented between towers. No pre-existing output is overwritten.
All candidate wall lines remain unreviewed and use image-pixel coordinates.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import mapping

from color_compare import local_lab_normalize
from extract import make_mask, read_image, save_png
from finalize import build_candidates, exclude_frame
from fortifications import analyze as fortification_baseline, ellipsoid
from shoreline import v07_mask
from wall_trace import (darkness_layers, thin_masonry, tower_seeds,
                        trace_tower_arms)


def building_roofs_v05(original, building_cfg):
    """Recreate v0.5's *accepted* roof candidates, not the raw color mask."""
    corrected = local_lab_normalize(
        original, building_cfg.get("v04_color_normalization", {}))
    base_mask = exclude_frame(make_mask(corrected, building_cfg),
                              building_cfg)
    objects, _, _ = build_candidates(
        base_mask, building_cfg["v02_filters"],
        float(building_cfg.get("simplify_fraction", .009)))
    accepted = np.zeros(original.shape[:2], dtype=np.uint8)
    for obj in objects:
        cv2.drawContours(accepted, [obj["contour"]], -1, 255, -1)
    return accepted, len(objects)


def expand_mask(mask, radius):
    radius = int(radius)
    if not 0 <= radius <= 20:
        raise ValueError("Mask padding must be 0..20 pixels")
    if not radius:
        return mask.copy()
    return cv2.dilate(mask, ellipsoid(radius))


def build_suppression(roofs, tower_anchors, attached_buildings, cfg):
    """Protect trusted tower pixels even when roof candidates overlap them.

    'Attached building' is only a provisional v0.21 classification; save a
    separate diagnostic for it so false preservation is easy to identify.
    """
    expanded = expand_mask(roofs, cfg.get("building_outline_pad_px", 3))
    protected_towers = expand_mask(
        tower_anchors, cfg.get("preserve_tower_radius_px", 5))
    if cfg.get("preserve_attached_buildings", True):
        protected_buildings = expand_mask(
            attached_buildings, cfg.get("attached_building_pad_px", 2))
    else:
        protected_buildings = np.zeros_like(roofs)
    protected = cv2.bitwise_or(protected_towers, protected_buildings)
    suppressed = cv2.bitwise_and(
        expanded, cv2.bitwise_not(protected))
    return {"accepted_roofs": roofs, "expanded_roofs": expanded,
            "protected_towers": protected_towers,
            "protected_attached_buildings": protected_buildings,
            "suppressed_building_outlines": suppressed}


def variants(near_black, gray, masks, cfg):
    """Baseline, hard removal and softer darkness penalty within roof halos."""
    if near_black.shape != gray.shape:
        raise ValueError("Gray and near-black images must be aligned")
    suppress = masks["suppressed_building_outlines"] > 0
    hard = near_black.copy()
    hard[suppress] = 0

    soft_black_max = int(cfg.get("soft_black_gray_max", 43))
    if not 0 <= soft_black_max <= 180:
        raise ValueError("soft_black_gray_max must be 0..180")
    soft = near_black.copy()
    soft[suppress & (gray > soft_black_max)] = 0

    return {"baseline_v022": near_black.copy(), "hard_mask": hard,
            "soft_darkness": soft}


def trace_variants(variant_masks, anchors, trace_cfg):
    results = {}
    for name, black in variant_masks.items():
        skeleton = thin_masonry(black)
        routes = trace_tower_arms(skeleton, anchors, trace_cfg)
        results[name] = {"skeleton": skeleton, "routes": routes}
    return results


def trace_pixels(shape, routes):
    mask = np.zeros(shape, dtype=np.uint8)
    for item in routes:
        coords = np.rint(np.asarray(item["line"].coords)).astype(np.int32)
        if len(coords) >= 2:
            cv2.polylines(mask, [coords], False, 255, 1)
    return mask


def render(image, anchors, routes):
    overlay = image.copy()
    for item in routes:
        coords = np.rint(np.asarray(item["line"].coords)).astype(np.int32)
        if len(coords) >= 2:
            color = ((30, 190, 30) if item["end_reason"] == "other_tower"
                     else (0, 205, 255))
            cv2.polylines(overlay, [coords], False, color, 2)
    overlay[anchors > 0] = (205, 35, 215)
    return overlay


def write_routes(path, routes, variant):
    features = []
    for index, route in enumerate(routes, 1):
        fid = f"W{index:04d}"
        features.append({
            "type": "Feature", "id": fid,
            "properties": {
                "id": fid, "classification": "wall_trace_candidate",
                "variant": variant, "review_status": "unreviewed",
                "source_tower_component": route["origin_component"],
                "termination": route["end_reason"],
                "coordinate_system": "image_pixels_top_left_origin",
            },
            "geometry": mapping(route["line"]),
        })
    path.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features},
        indent=2), encoding="utf-8")


def run(original, hdr, profile, building_cfg):
    if original.shape != hdr.shape:
        raise ValueError("Original and HDR photos must be pixel aligned")
    cfg = profile["fortifications_v023"]
    tracing = profile["fortifications_v022"]
    previous = fortification_baseline(
        hdr, profile, building_cfg, profile["fortifications_v021"])
    layers = darkness_layers(hdr, tracing, previous["analysis_region"])
    # Freeze tower anchors across all variants: otherwise comparisons reflect
    # changes in the seed selection as well as building suppression.
    water = v07_mask(original, profile)
    anchors, tower_info, rejected = tower_seeds(
        previous["tower_thickenings"], layers["near_black"], tracing, water)
    roofs, building_count = building_roofs_v05(original, building_cfg)
    masks = build_suppression(
        roofs, anchors, previous["large_attached_buildings"], cfg)
    masked = variants(layers["near_black"], layers["gray_value"], masks, cfg)
    traced = trace_variants(masked, anchors, tracing)
    return {
        "masks": masks, "variants": masked, "traced": traced,
        "tower_anchors": anchors, "tower_info": tower_info,
        "rejected_towers": rejected, "building_candidates_v05": building_count,
        "gray": layers["gray_value"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="v0.23 building-aware masonry tracing A/B experiment")
    parser.add_argument("--image", required=True, type=Path,
                        help="HDR map, unmodified by prior overlays")
    parser.add_argument("--original", required=True, type=Path,
                        help="Original map for v0.5 LAB roof recognition")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--building-config", type=Path,
                        default=Path("MapExtractor/sample-config.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/fortifications_v023"))
    args = parser.parse_args()
    hdr, original = read_image(args.image), read_image(args.original)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    buildings = json.loads(args.building_config.read_text(encoding="utf-8"))
    result = run(original, hdr, profile, buildings)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for name, mask in result["masks"].items():
        save_png(out / (name + ".png"), mask)
    for name, black in result["variants"].items():
        save_png(out / (name + "_black.png"), black)
    save_png(out / "tower_anchors_unchanged.png", result["tower_anchors"])

    overlay = hdr.copy()
    suppress = result["masks"]["suppressed_building_outlines"] > 0
    overlay[suppress] = (0, 170, 245)  # orange: suppressed house outlines
    protected = result["masks"]["protected_towers"] > 0
    overlay[protected] = (205, 35, 215)  # magenta: tower protection
    attached = result["masks"]["protected_attached_buildings"] > 0
    overlay[attached & ~protected] = (30, 190, 30)  # green: review preservation
    save_png(out / "house_suppression_overlay.png", overlay)

    counts = {}
    base_pixels = None
    for name, value in result["traced"].items():
        save_png(out / (name + "_skeleton.png"), value["skeleton"])
        save_png(out / (name + "_wall_overlay.png"),
                 render(hdr, result["tower_anchors"], value["routes"]))
        write_routes(out / (name + "_wall_traces.geojson"),
                     value["routes"], name)
        pixels = trace_pixels(hdr.shape[:2], value["routes"]) > 0
        if name == "baseline_v022":
            base_pixels = pixels
        else:
            diff = np.zeros_like(hdr)
            diff[base_pixels & pixels] = (170, 170, 170)
            diff[base_pixels & ~pixels] = (0, 0, 235)   # red: baseline-only
            diff[~base_pixels & pixels] = (0, 210, 0)   # green: new-only
            save_png(out / (name + "_trace_difference.png"), diff)
        counts[name] = {
            "black_pixels": int(np.count_nonzero(result["variants"][name])),
            "wall_fragments": len(value["routes"]),
            "total_trace_length_px": round(sum(
                item["line"].length for item in value["routes"]), 1),
            "end_reasons": {
                reason: sum(item["end_reason"] == reason
                            for item in value["routes"])
                for reason in sorted(set(
                    item["end_reason"] for item in value["routes"]))
            },
        }
    summary = {
        "version": "0.23",
        "status": "unreviewed_comparison",
        "tower_anchors_shared": len(result["tower_info"]),
        "building_candidates_v05": result["building_candidates_v05"],
        "suppressed_pixels": int(np.count_nonzero(
            result["masks"]["suppressed_building_outlines"])),
        "protected_tower_pixels": int(np.count_nonzero(
            result["masks"]["protected_towers"])),
        "variants": counts,
        "warning": "v0.21 attached-building candidates can be false positives. "
                   "House suppression can erase genuine wall ink; compare "
                   "baselines visually before adopting either variant.",
    }
    (out / "stats_v023.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
