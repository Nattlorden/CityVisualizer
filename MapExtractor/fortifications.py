"""Experimental fortification inventory, not a definitive wall reconstruction.

Four *candidate* layers: wall-like dark masonry, thickening/tower candidates,
possible gaps/gates, and large roof-like buildings touching the dark structure.
No Harn-specific coordinates are embedded in the detector.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, mapping

from extract import read_image, save_png, make_mask
from land_boundary_dual import ink_mask
from water import spatial_mask


def ellipsoid(radius):
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))


def region_mask(shape, cfg):
    mask = np.full(shape[:2], 255, np.uint8)
    if cfg.get("regions"):
        mask[:] = 0
        for region in cfg["regions"]:
            polygon = np.asarray(region, np.int32)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                raise ValueError("fortification region needs 3+ [x,y] vertices")
            cv2.fillPoly(mask, [polygon], 255)
    for region in cfg.get("exclude_regions", []):
        cv2.fillPoly(mask, [np.asarray(region, np.int32)], 0)
    return mask


def structural_ink(image, ink_cfg, cfg, region=None):
    """Find broad dark strokes, separate from fine text and hatching.

    Very dark pixels are kept even where the local contrast detector weakens;
    the morphological opening rejects isolated single-pixel print noise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = ink_mask(image, ink_cfg)
    ink |= ((gray < int(cfg.get("absolute_dark_max", 70))).astype(np.uint8)*255)
    opening = int(cfg.get("opening_radius_px", 2))
    if opening:
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, ellipsoid(opening))
    if region is not None:
        ink &= region
    return ink


def contours_as_features(binary, minimum, label, prefix):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum:
            continue
        eps = max(0.6, cv2.arcLength(contour, True)*.003)
        pts = cv2.approxPolyDP(contour, eps, True).reshape(-1,2)
        if len(pts) < 3:
            continue
        poly = Polygon(pts.tolist())
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        fid = f"{prefix}{len(features)+1:04d}"
        features.append({"type":"Feature","id":fid,
                         "properties":{"id":fid,"classification":label,
                                       "review_status":"unreviewed",
                                       "area_px2":round(area,1),
                                       "coordinate_system":"image_pixels_top_left_origin"},
                         "geometry":mapping(poly)})
    return features


def analyze(image, profile, buildings_config, cfg):
    region = region_mask(image.shape, cfg)
    structure = structural_ink(image, profile["land_boundary_v012"], cfg,
                               region)
    # Distance-to-background is local half-thickness of a dark stroke.
    thickness = cv2.distanceTransform((structure>0).astype(np.uint8),
                                      cv2.DIST_L2, 3)
    thin_start = float(cfg.get("wall_min_halfwidth_px", 1.5))
    thick_start = float(cfg.get("tower_min_halfwidth_px", 4.5))
    # A thick core is a potential tower or foundation. Classifying it as
    # tower requires human context (wall attachment, layout, shape).
    wall_core = (thickness >= thin_start).astype(np.uint8)*255
    tower_core = (thickness >= thick_start).astype(np.uint8)*255
    # Restore the original stroke width for the broad structural proposal.
    wall_candidate = cv2.dilate(wall_core, ellipsoid(
        cfg.get("restore_wall_radius_px", 2))) & structure
    tower_candidate = cv2.dilate(tower_core, ellipsoid(
        cfg.get("restore_tower_radius_px", 4))) & structure
    # Big roofed structures adjacent to proposed masonry may be part of the
    # fortification. This is ONLY topological adjacency, not proof of use.
    roofs = make_mask(image, buildings_config) & region
    roof_contours, _ = cv2.findContours(roofs, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    wall_near = cv2.dilate(wall_candidate, ellipsoid(
        cfg.get("building_wall_proximity_px", 8))) > 0
    buildings = np.zeros_like(roofs)
    min_building = float(cfg.get("large_building_area_px2", 250))
    for contour in roof_contours:
        if cv2.contourArea(contour) < min_building:
            continue
        isolated = np.zeros_like(roofs)
        cv2.drawContours(isolated, [contour], -1, 255, -1)
        edge = cv2.morphologyEx(isolated, cv2.MORPH_GRADIENT,
                               np.ones((3,3), np.uint8)) > 0
        if np.count_nonzero(edge & wall_near) >= int(
                cfg.get("min_wall_contact_pixels", 5)):
            cv2.drawContours(buildings, [contour], -1, 255, -1)
    # Gate proposals: narrow light gaps between two otherwise dark pieces
    # within the same small neighborhood. These are weak hypotheses.
    close = cv2.morphologyEx(wall_candidate, cv2.MORPH_CLOSE,
                            ellipsoid(cfg.get("gate_max_half_gap_px", 4)))
    gaps = (close & cv2.bitwise_not(wall_candidate)) & region
    # Avoid calling a gap through a building a gate.
    gaps &= cv2.bitwise_not(roofs)
    # Filter to gaps with sufficiently many dark neighbors.
    neighbors = cv2.dilate(wall_candidate, ellipsoid(3))
    gaps &= neighbors
    return {"structural_ink":structure, "wall_candidates":wall_candidate,
            "tower_thickenings":tower_candidate,
            "large_attached_buildings":buildings,
            "possible_gaps":gaps, "analysis_region":region}


def main():
    p=argparse.ArgumentParser(description="Fortification candidates and diagnostics")
    p.add_argument("--image",required=True,type=Path)
    p.add_argument("--profile",required=True,type=Path,
                   help="Existing water/ink profile, used for HDR ink configuration")
    p.add_argument("--building-config",type=Path,
                   default=Path("MapExtractor/sample-config.json"))
    p.add_argument("--output",type=Path,
                   default=Path("output/fortifications_v021"))
    args=p.parse_args()
    image=read_image(args.image)
    profile=json.loads(args.profile.read_text(encoding="utf-8"))
    buildings=json.loads(args.building_config.read_text(encoding="utf-8"))
    cfg=profile["fortifications_v021"]
    masks=analyze(image,profile,buildings,cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    colors=[("wall_candidates",(245,140,30)),    # blue in BGR
            ("tower_thickenings",(200,30,210)), # magenta
            ("possible_gaps",(0,200,245)),       # yellow
            ("large_attached_buildings",(30,190,30))] # green
    overlay=image.copy()
    for name,color in colors:
        overlay[masks[name]>0]=color
    for name,mask in masks.items():
        save_png(args.output/(name+".png"),mask)
    save_png(args.output/"fortification_candidates_overlay.png",overlay)
    all_features=[]
    for layer,prefix,minimum in (
        ("wall_candidates","W",float(cfg.get("wall_min_area_px2",20))),
        ("tower_thickenings","T",float(cfg.get("tower_min_area_px2",18))),
        ("possible_gaps","G",float(cfg.get("gate_min_gap_area_px2",3))),
        ("large_attached_buildings","B",float(cfg.get("large_building_area_px2",250)))):
        features=contours_as_features(masks[layer], minimum,
            {"wall_candidates":"wall_segment_candidate",
             "tower_thickenings":"tower_or_bastion_candidate",
             "possible_gaps":"possible_wall_gap_not_verified_gate",
             "large_attached_buildings":"possible_fortification_building"}[layer],
             prefix)
        all_features.extend(features)
        (args.output/(layer+".geojson")).write_text(json.dumps(
            {"type":"FeatureCollection","features":features},indent=2),
            encoding="utf-8")
    stats={"version":"0.21","unreviewed":True,
           "counts":{key:sum(f["properties"]["classification"]==kind for f in all_features)
                     for key,kind in [
                       ("walls","wall_segment_candidate"),
                       ("towers","tower_or_bastion_candidate"),
                       ("gaps","possible_wall_gap_not_verified_gate"),
                       ("large_attached_buildings","possible_fortification_building")]},
           "warning":"All dark lines can contain streets, text and house outlines. "
                     "A gap is not a confirmed gate and a thickening not a confirmed tower."}
    (args.output/"stats_v021.json").write_text(
        json.dumps(stats,indent=2),encoding="utf-8")
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
