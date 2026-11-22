"""CityVisualizer v0.8: experimental narrow-band shoreline refinement.

Reuses v0.7. Map-specific parameters belong in the water profile.
Outputs unreviewed water polygons and shoreline lines in image pixels.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import LineString, mapping
from extract import read_image, save_png
from water import make_water_mask, spatial_mask, find_water_candidates
from water_refine import grow_near_water, seed_lakes


def v07_mask(image, profile):
    base = make_water_mask(image, profile)
    permitted = spatial_mask(image.shape, profile)
    near = grow_near_water(image, base, profile, permitted)
    seeds, _ = seed_lakes(image, profile, permitted)
    return cv2.bitwise_and(base | near | seeds, permitted)


def refine(image, base, allowed, opts):
    band = int(opts.get("band_px", 5))
    iterations = int(opts.get("iterations", 3))
    if not 2 <= band <= 30 or not 1 <= iterations <= 8:
        raise ValueError("shoreline band_px must be 2..30 and iterations 1..8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*band+1, 2*band+1))
    interior = cv2.erode(base, kernel) > 0
    possible = cv2.dilate(base, kernel) > 0
    labels = np.full(base.shape, cv2.GC_BGD, np.uint8)
    labels[possible & (allowed > 0)] = cv2.GC_PR_BGD
    labels[base > 0] = cv2.GC_PR_FGD
    labels[interior] = cv2.GC_FGD
    labels[allowed == 0] = cv2.GC_BGD
    bg = np.zeros((1,65), np.float64)
    fg = np.zeros((1,65), np.float64)
    cv2.grabCut(image, labels, None, bg, fg, iterations, cv2.GC_INIT_WITH_MASK)
    result = np.where(np.isin(labels,(cv2.GC_FGD,cv2.GC_PR_FGD)),255,0).astype(np.uint8)
    result[interior] = 255
    result[allowed == 0] = 0
    return result


def boundary(mask):
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3,3),np.uint8))


def shoreline_features(mask, opts):
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    features = []
    for idx, c in enumerate(contours):
        if cv2.arcLength(c, True) < float(opts.get("min_length_px", 20)):
            continue
        points = cv2.approxPolyDP(c,float(opts.get("simplify_px",0.9)),True).reshape(-1,2)
        if len(points) < 3:
            continue
        coordinates = points.tolist() + [points[0].tolist()]
        ident = f"S{len(features)+1:04d}"
        features.append({"type":"Feature", "id":ident,
                         "properties":{"id":ident,"classification":"shoreline_candidate",
                                       "review_status":"unreviewed",
                                       "boundary_role":"interior" if hierarchy[0][idx][3]>=0 else "exterior",
                                       "coordinate_system":"image_pixels_top_left_origin",
                                       "confidence":None},
                         "geometry":mapping(LineString(coordinates))})
    return features


def main():
    parser = argparse.ArgumentParser(description="Compare original v0.7 and experimental shoreline")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/shoreline_v08"))
    args = parser.parse_args()
    img = read_image(args.image)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    opts = profile.get("shoreline", {})
    original = v07_mask(img,profile)
    water = refine(img,original,spatial_mask(img.shape,profile),opts)
    old_edge, new_edge = boundary(original),boundary(water)
    added = (water>0)&(original==0)
    removed = (water==0)&(original>0)
    overlay = img.copy()
    overlay[added] = (50,220,50)
    overlay[removed] = (180,45,220)
    overlay = cv2.addWeighted(img,0.52,overlay,0.48,0)
    overlay[new_edge>0] = (0,150,250)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,int(opts.get("canny_low",40)),int(opts.get("canny_high",110)))
    band = int(opts.get("band_px",5))
    corridor = cv2.dilate(old_edge,np.ones((2*band+1,2*band+1),np.uint8))
    evidence = cv2.bitwise_and(edges,corridor)
    candidates, total = find_water_candidates(water,profile)
    water_features = []
    for i, (_,area,geom,_) in enumerate(candidates,1):
        ident = f"W{i:04d}"
        water_features.append({"type":"Feature","id":ident,
                               "properties":{"id":ident,"classification":"water_candidate",
                                             "review_status":"unreviewed",
                                             "source_method":"shoreline_v08",
                                             "area_px2":round(area,1),
                                             "coordinate_system":"image_pixels_top_left_origin"},
                               "geometry":mapping(geom)})
    path = args.output
    path.mkdir(parents=True,exist_ok=True)
    for name, data in (("water_mask_v07_baseline.png",original),
                       ("water_mask_v08_experimental.png",water),
                       ("shoreline_overlay_v08.png",overlay),
                       ("shoreline_dark_line_evidence_v08.png",evidence),
                       ("shoreline_edge_v08.png",new_edge)):
        save_png(path/name,data)
    def write(name, obj):
        (path/name).write_text(json.dumps(obj,indent=2),encoding="utf-8")
    shore_features = shoreline_features(water,opts)
    write("shoreline_candidates_v08.geojson",{"type":"FeatureCollection","features":shore_features})
    write("water_candidates_v08.geojson",{"type":"FeatureCollection","features":water_features})
    stats = {"version":"0.8","status":"experimental_unreviewed",
             "water_pixels_v07":int(np.count_nonzero(original)),
             "water_pixels_v08":int(np.count_nonzero(water)),
             "added_pixels":int(np.count_nonzero(added)),
             "removed_pixels":int(np.count_nonzero(removed)),
             "shoreline_candidates":len(shore_features),
             "water_candidates":len(water_features),"raw_water_contours":total}
    write("shoreline_stats_v08.json",stats)
    print(json.dumps(stats,indent=2))
    print(f"Saved: {path.resolve()}")


if __name__=="__main__":
    main()
