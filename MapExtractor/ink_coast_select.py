"""v0.14 experimental coastal ink selection by opposite-side color evidence.

Preserves the original HDR ink mask and the v0.7 outer water mask.
Selects only dark line pixels with green-land evidence on one side and
shore-colored evidence on the other. Connected coast segments and islands
remain separate; no inferred coastal polygon is created.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import mapping

from extract import read_image, save_png
from land_boundary_dual import ink_mask, color_mask
from ink_connect import skeletonize, propose_bridges, trace_chains
from shoreline import v07_mask
from water import spatial_mask


def color_fraction(mask, sigma):
    return cv2.GaussianBlur(
        (mask > 0).astype(np.float32), (0, 0), sigma)


def side_score(hdr, ink, water, allowed, cfg):
    """Return pixel-level inner-shore evidence, without following region borders.

    Gradients of *broad green-color density* supply the line normal. Two
    offset samples probe opposite sides of the actual ink, keeping its pixels
    as the positional source instead of moving the line to a color contour.
    """
    green = color_mask(hdr, cfg["green_hsv_ranges"])
    sand = color_mask(hdr, cfg["shore_hsv_ranges"])
    sigma = float(cfg.get("color_smoothing_sigma_px", 5))
    normal_sigma = float(cfg.get("normal_sigma_px", 11))
    offset = float(cfg.get("side_offset_px", 11))
    if min(sigma, normal_sigma, offset) <= 0:
        raise ValueError("Smoothing sigma and offset must be positive")
    g = color_fraction(green, sigma)
    s = color_fraction(sand, sigma)
    # Broad-scale gradient points into green land; avoid gradients around
    # individual 1-2 px map symbols.
    broad = color_fraction(green, normal_sigma)
    dx = cv2.Sobel(broad, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(broad, cv2.CV_32F, 0, 1, ksize=3)
    strength = np.sqrt(dx*dx + dy*dy)
    nx = dx / np.maximum(strength, 1e-6)
    ny = dy / np.maximum(strength, 1e-6)
    h, w = ink.shape
    xx = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w))
    yy = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w))
    def sample(field, side):
        return cv2.remap(field,
            xx + side*offset*nx, yy + side*offset*ny,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)
    gp, gm = sample(g, 1), sample(g, -1)
    sp, sm = sample(s, 1), sample(s, -1)
    # Allow either normal orientation.
    plus_land = np.minimum(gp, sm)
    minus_land = np.minimum(gm, sp)
    score = np.maximum(plus_land, minus_land)
    land_side = np.where(plus_land >= minus_land, gp, gm)
    shore_side = np.where(plus_land >= minus_land, sm, sp)
    other_land = np.where(plus_land >= minus_land, gm, gp)
    other_shore = np.where(plus_land >= minus_land, sp, sm)
    separation = (land_side-other_land) + (shore_side-other_shore)
    # v0.7 is a maximum-distance guard only; the inner line is permitted
    # far from the water in naturally broad sand/stone beach zones.
    water_distance = cv2.distanceTransform(
        (water == 0).astype(np.uint8), cv2.DIST_L2, 3)
    max_distance = float(cfg.get("max_water_distance_px", 150))
    core = ((ink > 0) & (allowed > 0) &
            (water_distance <= max_distance) &
            (strength >= float(cfg.get("min_normal_strength", 0.045))) &
            (score >= float(cfg.get("min_side_color_fraction", 0.22))) &
            (separation >= float(cfg.get("min_side_separation", 0.16))))
    # Do not export linework on the already-detected water side.
    core &= (water == 0)
    return core.astype(np.uint8)*255, {
        "green_color": green,
        "shore_color": sand,
        "side_score": np.uint8(np.clip(score*255,0,255)),
        "side_separation": np.uint8(np.clip((separation+1)*85,0,255)),
        "water_distance": np.uint8(np.clip(water_distance,0,255)),
    }


def filter_components(mask, cfg):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    result = np.zeros_like(mask)
    minimum = int(cfg.get("min_component_pixels", 12))
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= minimum:
            result[labels == i] = 255
    return result


def run(original, hdr, profile, cfg):
    if original.shape != hdr.shape:
        raise ValueError("Original and HDR must be pixel-aligned and identical dimensions")
    permitted = spatial_mask(original.shape, profile)
    water = v07_mask(original, profile)
    ink = ink_mask(hdr, profile["land_boundary_v012"]) & permitted
    side, diagnostics = side_score(hdr, ink, water, permitted, cfg)
    skeleton = skeletonize(side)
    corridor = cv2.dilate(side, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2*int(cfg.get("bridge_corridor_px", 8))+1,)*2)) > 0
    bridges, proposed, endpoints = propose_bridges(skeleton, corridor, cfg)
    connected = cv2.bitwise_or(skeleton, bridges)
    selected = filter_components(connected, cfg)
    masks = {
        "ink_hdr_raw": ink,
        "side_evidence": side,
        "side_skeleton": skeleton,
        "bridges": bridges,
        "connected": connected,
        "land_boundary_candidates": selected,
        "water_v07_reference": water,
        **diagnostics,
    }
    return masks, proposed, endpoints


def main():
    parser = argparse.ArgumentParser(description="v0.14 land/shore two-sided ink selector")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--hdr", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/ink_coast_v014"))
    args = parser.parse_args()
    original, hdr = read_image(args.image), read_image(args.hdr)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    cfg = profile["ink_coast_v014"]
    masks, links, endpoints = run(original, hdr, profile, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        save_png(args.output / (name+".png"), mask)
    for image, name in ((hdr, "hdr"), (original, "original")):
        result = image.copy()
        result[masks["land_boundary_candidates"]>0] = (0,0,245)
        result[masks["bridges"]>0] = (220,30,215)
        save_png(args.output / (name+"_coast_overlay.png"), result)
    # Diagnostic: what the algorithm chose against the *complete* dark ink.
    inspection = hdr.copy()
    inspection[masks["ink_hdr_raw"]>0] = (120,130,130)
    inspection[masks["land_boundary_candidates"]>0] = (0,0,245)
    inspection[masks["bridges"]>0] = (220,30,215)
    save_png(args.output/"selection_vs_all_ink.png", inspection)
    features = []
    for i, line in enumerate(trace_chains(
            masks["land_boundary_candidates"],
            float(cfg.get("min_trace_length_px", 8))),1):
        fid = f"LB{i:04d}"
        features.append({"type":"Feature","id":fid,
            "properties":{"id":fid,"classification":"inner_land_shore_candidate",
                          "review_status":"unreviewed",
                          "coordinate_system":"image_pixels_top_left_origin"},
            "geometry":mapping(line)})
    (args.output/"land_boundary_candidates_v014.geojson").write_text(
        json.dumps({"type":"FeatureCollection","features":features},indent=2),
        encoding="utf-8")
    (args.output/"bridge_proposals_v014.json").write_text(
        json.dumps([{"from":a,"to":b,"score":s} for a,b,s in links],indent=2),
        encoding="utf-8")
    report={"version":"0.14",
            "raw_ink_pixels":int(np.count_nonzero(masks["ink_hdr_raw"])),
            "two_sided_evidence_pixels":int(np.count_nonzero(masks["side_evidence"])),
            "selected_pixels":int(np.count_nonzero(masks["land_boundary_candidates"])),
            "bridge_count":len(links),"endpoint_candidates":endpoints,
            "geojson_lines":len(features),
            "limitations":"Strict green/shore color contrast misses quay, cliffs, "
                          "bare rock and partially occluded contours. Manual review required."}
    (args.output/"stats_v014.json").write_text(json.dumps(report,indent=2),
                                               encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
