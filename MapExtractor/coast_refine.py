"""CityVisualizer v0.16: conservative cleanup/classification of v0.15 coast.

Keeps v0.15 intact. No water-mask modification, no fabricated bay geometry.
Outputs separate natural/built/uncertain candidate segments with explicit reasons.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, mapping

from coast_guided import run as guided_run
from extract import read_image, save_png
from land_boundary_dual import color_mask


def consecutive_clean(coords):
    """Discard consecutive duplicate coordinates, preserving path order."""
    points = np.asarray(coords, dtype=float)
    if len(points) < 2:
        return points
    keep = np.concatenate(([True], np.any(np.diff(points, axis=0) != 0, axis=1)))
    return points[keep]


def remove_spikes(coords, options):
    """Only prune unmistakable backtracking detours with short endpoints.

    Do not smooth arbitrary bends: natural headlands and bays must survive.
    """
    points = consecutive_clean(coords)
    maximum = float(options.get("spike_span_px", 7))
    detour_ratio = float(options.get("spike_detour_ratio", 2.8))
    removed = 0
    if len(points) < 4:
        return points, removed
    # Conservative single pass; do not use Douglas-Peucker to erase bays.
    output = [points[0]]
    i = 1
    while i < len(points)-1:
        best = None
        for j in range(min(len(points)-1, i+3), i, -1):
            straight = float(np.linalg.norm(points[j]-output[-1]))
            if straight > maximum:
                continue
            path = np.vstack([output[-1], points[i:j+1]])
            distance = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if distance >= maximum and distance > detour_ratio * max(straight, 0.5):
                best = j
                break
        if best is not None:
            removed += best-i
            i = best
        output.append(points[i])
        i += 1
    output.append(points[-1])
    return consecutive_clean(output), removed


def local_fraction(binary, xs, ys, radius):
    """Fraction of marked pixels in a patch, sampled at segment points."""
    integral = cv2.integral((binary > 0).astype(np.uint8))
    h, w = binary.shape
    x1 = np.maximum(xs-radius, 0); y1 = np.maximum(ys-radius, 0)
    x2 = np.minimum(xs+radius+1, w); y2 = np.minimum(ys+radius+1, h)
    count = (integral[y2, x2]-integral[y1, x2]
             -integral[y2, x1]+integral[y1, x1])
    return count / np.maximum((x2-x1)*(y2-y1), 1)


def segment_evidence(line, hdr, profile, opts, water):
    """Classify landscape context without moving coastline coordinates."""
    points = np.rint(np.asarray(line.coords)).astype(int)
    h, w = water.shape
    xs = np.clip(points[:, 0], 0, w-1); ys = np.clip(points[:, 1], 0, h-1)
    green = color_mask(hdr, opts["natural_hsv_ranges"])
    built = color_mask(hdr, opts["built_hsv_ranges"])
    radius = int(opts.get("context_radius_px", 10))
    if radius < 1 or radius > 50:
        raise ValueError("context_radius_px must be 1..50")
    natural = float(np.mean(local_fraction(green, xs, ys, radius)))
    urban = float(np.mean(local_fraction(built, xs, ys, radius)))
    # Evidence is deliberately local context, NOT a conclusive landcover class.
    if natural >= float(opts.get("natural_min_fraction", 0.18)) and natural > urban*float(opts.get("type_margin",1.25)):
        kind = "natural_candidate"
    elif urban >= float(opts.get("built_min_fraction", 0.22)) and urban > natural*float(opts.get("type_margin",1.25)):
        kind = "built_candidate"
    else:
        kind = "unknown"
    return kind, round(natural, 3), round(urban, 3)


def curvature_score(line, options):
    pts = np.asarray(line.coords, dtype=float)
    if len(pts) < 3:
        return 0.0
    length = float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())
    chord = float(np.linalg.norm(pts[-1]-pts[0]))
    return round(length / max(chord, float(options.get("min_chord_px", 4))), 3)


def refine(original, hdr, profile, cfg, mask_override=None):
    guided = dict(profile["coast_guided_v015"])
    # Run baseline strictly unmodified, retaining one-to-one candidate origin.
    masks, lines, baseline_stats = guided_run(
        original, hdr, profile, guided, mask_override)
    items = []
    rejected = []
    for index, item in enumerate(lines,1):
        raw = item["line"]
        coords, pruned = remove_spikes(raw.coords, cfg)
        if len(coords) < 2:
            rejected.append({"source_id":index,"reason":"collapsed_spike"})
            continue
        line = LineString(coords)
        if not line.is_valid or line.length == 0:
            rejected.append({"source_id":index,"reason":"invalid_line"})
            continue
        kind, green, built = segment_evidence(
            line, hdr, profile, cfg, masks["water_v07"])
        tortuosity = curvature_score(line, cfg)
        short = line.length < float(cfg.get("min_reliable_length_px", 28))
        extreme = tortuosity > float(cfg.get("winding_review_threshold", 4))
        suspicious_built = kind == "built_candidate" and (
            line.length < float(cfg.get("built_review_length_px", 65)))
        reasons = []
        if short: reasons.append("short_fragment")
        if extreme: reasons.append("winding_or_bay_review")
        if suspicious_built: reasons.append("possible_quay_spur")
        if kind == "unknown": reasons.append("landcover_uncertain")
        if pruned: reasons.append("short_backtrack_removed")
        status = "review" if reasons else "candidate"
        items.append({"source_id":index, "line":line,
                      "boundary_kind": "island_or_land_hole" if item["water_hole"]
                                       else "water_exterior",
                      "type":kind, "confidence": "low" if (short or extreme or suspicious_built)
                            else ("medium" if kind == "unknown" else "provisional_high"),
                      "review_status":status, "reasons":reasons,
                      "natural_fraction":green, "built_fraction":built,
                      "winding_ratio":tortuosity, "pruned_points":pruned})
    return masks, items, rejected, baseline_stats


def main():
    p = argparse.ArgumentParser(description="v0.16 v0.15 cleanup with terrain confidence")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--hdr", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--water-mask", type=Path)
    p.add_argument("--output", type=Path, default=Path("output/coast_refined_v016"))
    args = p.parse_args()
    original, hdr = read_image(args.image), read_image(args.hdr)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    cfg = profile["coast_refine_v016"]
    override = None
    if args.water_mask is not None:
        override = cv2.imread(str(args.water_mask), cv2.IMREAD_GRAYSCALE)
        if override is None or override.shape != original.shape[:2]:
            raise ValueError("Water-mask must exist and match image dimensions")
        override = np.where(override > 127, 255, 0).astype(np.uint8)
    masks, items, rejected, base = refine(original, hdr, profile, cfg, override)
    folder = args.output
    folder.mkdir(parents=True,exist_ok=True)
    # Preserve v0.15 result verbatim for visual A/B comparisons.
    baseline = hdr.copy()
    original_image = original.copy()
    result = hdr.copy()
    edge = cv2.morphologyEx(masks["water_v07"], cv2.MORPH_GRADIENT,
                             np.ones((3,3),np.uint8))
    baseline[edge>0] = (240,130,20)
    result[edge>0] = (240,130,20)
    colors = {"natural_candidate":(20,180,40),
              "built_candidate":(255,155,35), "unknown":(0,0,230)}
    review_mask = np.zeros(masks["water_v07"].shape,np.uint8)
    features = []
    for n, obj in enumerate(items,1):
        coords = np.rint(np.asarray(obj["line"].coords)).astype(np.int32)
        if len(coords) < 2: continue
        ink_color = colors[obj["type"]]
        cv2.polylines(result,[coords],False,ink_color,2)
        if obj["review_status"] == "review":
            cv2.polylines(review_mask,[coords],False,255,2)
        fid = f"CR{n:04d}"
        props = {k:v for k,v in obj.items() if k!="line"}
        props.update({"id":fid,"layer":"inner_land_boundary_candidate",
                      "coordinate_system":"image_pixels_top_left_origin"})
        features.append({"type":"Feature","id":fid,"properties":props,
                         "geometry":mapping(obj["line"])})
    for obj in items:
        if obj["review_status"]=="review":
            continue
        pts = np.rint(np.asarray(obj["line"].coords)).astype(np.int32)
        cv2.polylines(original_image,[pts],False,colors[obj["type"]],2)
    for obj in items:
        if obj["review_status"]=="review":
            pts=np.rint(np.asarray(obj["line"].coords)).astype(np.int32)
            cv2.polylines(original_image,[pts],False,(0,0,230),2)
    for obj in items:
        pts=np.rint(np.asarray(obj["line"].coords)).astype(np.int32)
        cv2.polylines(baseline,[pts],False,(0,0,230),1)
    save_png(folder/"hdr_refined_overlay.png",result)
    save_png(folder/"original_refined_overlay.png",original_image)
    save_png(folder/"review_required.png",review_mask)
    save_png(folder/"hdr_candidate_comparison.png",baseline)
    save_png(folder/"water_v07_unchanged.png",masks["water_v07"])
    (folder/"coast_candidates_v016.geojson").write_text(
        json.dumps({"type":"FeatureCollection","features":features},indent=2),
        encoding="utf-8")
    summary = {"version":"0.16","baseline_segments":len(items)+len(rejected),
        "refined_segments":len(items),"rejected_invalid":len(rejected),
        "natural":sum(x["type"]=="natural_candidate" for x in items),
        "built":sum(x["type"]=="built_candidate" for x in items),
        "unknown":sum(x["type"]=="unknown" for x in items),
        "review":sum(x["review_status"]=="review" for x in items),
        "v015":base,
        "notes":["No v0.7 changes; no claimed bay recovery.",
                 "Natural/built is a preliminary context label, not proven material.",
                 "Red review features include potential quay spurs and winding bays."]}
    (folder/"stats_v016.json").write_text(json.dumps(summary,indent=2),
                                          encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
