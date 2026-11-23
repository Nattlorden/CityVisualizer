"""v0.20: last conservative coastal candidate experiment before freezing v0.16.

Keep the v0.16 candidate geometry as the input. On a copy, put segments
associated with small *independent* water components in a reversible review
layer; remove only very short unsupported inland-pointing triangular spurs.
Never modify v0.7, the source photograph, or earlier version files.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, mapping

from coast_refine import refine
from coast_diagnose import component_catalog, ink_support
from extract import read_image, save_png


def dense_points(line, step=2):
    coords = np.asarray(line.coords, dtype=float)
    if len(coords) < 2:
        return np.empty((0,2),dtype=int)
    result = []
    for a,b in zip(coords[:-1], coords[1:]):
        steps = max(2, int(np.linalg.norm(b-a)/step)+1)
        result.extend(np.linspace(a,b,steps)[:-1])
    result.append(coords[-1])
    return np.rint(np.asarray(result)).astype(int)


def small_water_proximity(water, cfg):
    labels, components = component_catalog(water, {
        "small_water_area_px": cfg.get("small_water_max_area_px", 2500)})
    selected = np.zeros_like(water)
    for comp in components:
        if comp["small"] and not comp["touches_image_edge"] and (
                comp["area_px"] >= int(cfg.get("small_water_min_area_px", 24))):
            selected[labels == comp["id"]] = 255
    radius = int(cfg.get("small_water_near_px", 17))
    if not 0 <= radius <= 80:
        raise ValueError("small_water_near_px must be between 0 and 80")
    near = selected if radius == 0 else cv2.dilate(
        selected,cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    return selected, near, components


def candidate_near_small_water(line, neighborhood, cfg):
    if line.length > float(cfg.get("small_water_review_max_line_px", 130)):
        return False
    pts = dense_points(line)
    if not len(pts):
        return False
    h,w=neighborhood.shape
    inside=(pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)
    if not np.any(inside):
        return False
    fraction=float(np.mean(neighborhood[pts[inside,1],
                                        pts[inside,0]]>0))
    return fraction >= float(cfg.get("small_water_near_fraction",0.7))


def water_distance(water):
    return cv2.distanceTransform((water==0).astype(np.uint8),
                                 cv2.DIST_L2,3)


def reject_short_spur(line, water_dist, ink, cfg):
    """Remove at most a local apex if both segments lack ink and it goes inland.

    The original full line remains available in the v0.16 reference.
    """
    coords = np.asarray(line.coords, dtype=float)
    if len(coords) < 3:
        return line, []
    maximum_base = float(cfg.get("triangle_max_base_px",14))
    minimum_height = float(cfg.get("triangle_min_height_px",6))
    minimum_inland = float(cfg.get("triangle_min_inland_px",4))
    min_ink = float(cfg.get("triangle_max_ink_support",0.18))
    kept=[coords[0]]
    changes=[]
    h,w=water_dist.shape
    for i in range(1,len(coords)-1):
        a,b,c=kept[-1],coords[i],coords[i+1]
        base_vec=c-a
        base=float(np.linalg.norm(base_vec))
        if base <= 1e-6 or base>maximum_base:
            kept.append(b)
            continue
        tip_height=abs(float(base_vec[0]*(b[1]-a[1]) -
                             base_vec[1]*(b[0]-a[0]))) / base
        p=np.rint(np.vstack([a,b,c])).astype(int)
        if np.any(p[:,0]<0) or np.any(p[:,0]>=w) or np.any(p[:,1]<0) or np.any(p[:,1]>=h):
            kept.append(b)
            continue
        distances=water_dist[p[:,1],p[:,0]]
        inland=float(distances[1]-max(distances[0],distances[2]))
        support=max(ink_support(a,b,ink,1),ink_support(b,c,ink,1))
        if (tip_height>=minimum_height and inland>=minimum_inland
                and support<min_ink):
            changes.append({"removed_apex":[float(x) for x in b],
                            "height_px":round(tip_height,2),
                            "inland_excursion_px":round(inland,2),
                            "ink_support":round(support,3)})
        else:
            kept.append(b)
    kept.append(coords[-1])
    result=LineString(kept) if len(kept)>=2 else line
    return result,changes


def process(items, water, ink, cfg):
    small, near, components = small_water_proximity(water,cfg)
    dist=water_distance(water)
    accepted,review,changes=[],[],[]
    for obj in items:
        record={key:value for key,value in obj.items() if key!="line"}
        line=obj["line"]
        if candidate_near_small_water(line,near,cfg):
            review.append({**record,"line":line,
                           "review_reason":"near_independent_small_water"})
            continue
        corrected,spikes=reject_short_spur(line,dist,ink,cfg)
        for s in spikes:
            changes.append({"source_id":obj["source_id"],**s})
        accepted.append({**record,"line":corrected,
                         "removed_apices":len(spikes)})
    return accepted,review,changes,small,components


def features(items, prefix, category):
    result=[]
    for index,obj in enumerate(items,1):
        fid=f"{prefix}{index:04d}"
        properties={k:v for k,v in obj.items() if k!="line"}
        properties.update({"id":fid,"layer":category,
                            "coordinate_system":"image_pixels_top_left_origin"})
        result.append({"type":"Feature","id":fid,
                       "properties":properties,"geometry":mapping(obj["line"])})
    return {"type":"FeatureCollection","features":result}


def main():
    parser=argparse.ArgumentParser(description="v0.20 conservative coast checkpoint")
    parser.add_argument("--image",required=True,type=Path)
    parser.add_argument("--hdr",required=True,type=Path)
    parser.add_argument("--profile",required=True,type=Path)
    parser.add_argument("--water-mask",type=Path)
    parser.add_argument("--output",type=Path,
                        default=Path("output/coast_checkpoint_v020"))
    args=parser.parse_args()
    original,hdr=read_image(args.image),read_image(args.hdr)
    profile=json.loads(args.profile.read_text(encoding="utf-8"))
    cfg=profile["coast_checkpoint_v020"]
    override=None
    if args.water_mask:
        override=cv2.imread(str(args.water_mask),cv2.IMREAD_GRAYSCALE)
        if override is None or override.shape!=original.shape[:2]:
            raise ValueError("Saved v0.7 mask not found or size mismatch")
        override=(override>127).astype(np.uint8)*255
    masks,items,_,_=refine(
        original,hdr,profile,profile["coast_refine_v016"],override)
    accepted,review,changes,small,components=process(
        items,masks["water_v07"],masks["ink_hdr_raw"],cfg)
    folder=args.output;folder.mkdir(parents=True,exist_ok=True)
    edge=cv2.morphologyEx(masks["water_v07"],cv2.MORPH_GRADIENT,
                         np.ones((3,3),np.uint8))
    reference=hdr.copy();proposal=hdr.copy();review_image=hdr.copy()
    for canvas in (reference,proposal,review_image):
        canvas[edge>0]=(240,130,20)
    for obj in items:
        pts=np.rint(np.asarray(obj["line"].coords)).astype(np.int32)
        cv2.polylines(reference,[pts],False,(0,0,235),1)
    for obj in accepted:
        pts=np.rint(np.asarray(obj["line"].coords)).astype(np.int32)
        cv2.polylines(proposal,[pts],False,(0,0,235),1)
        cv2.polylines(review_image,[pts],False,(0,0,235),1)
    for obj in review:
        pts=np.rint(np.asarray(obj["line"].coords)).astype(np.int32)
        cv2.polylines(review_image,[pts],False,(0,200,245),2)
    save_png(folder/"hdr_reference_v016.png",reference)
    save_png(folder/"hdr_candidate_v020.png",proposal)
    save_png(folder/"hdr_review_v020.png",review_image)
    save_png(folder/"small_water_components.png",small)
    save_png(folder/"water_v07_unchanged.png",masks["water_v07"])
    for name,records,prefix,category in (
        ("coast_candidates_v020.geojson",accepted,"LB","coast_candidate"),
        ("review_small_water_v020.geojson",review,"RV","coast_review")):
        (folder/name).write_text(json.dumps(features(records,prefix,category),
                                           indent=2),encoding="utf-8")
    (folder/"triangle_changes_v020.json").write_text(
        json.dumps(changes,indent=2),encoding="utf-8")
    report={"version":"0.20","baseline_segments":len(items),
            "candidate_segments":len(accepted),
            "small_water_review_segments":len(review),
            "removed_triangle_apices":len(changes),
            "small_connected_water_features":int(sum(
                x["small"] and not x["touches_image_edge"] for x in components)),
            "checkpoint":"Freeze v0.16 unless side-by-side image review proves better",
            "notes":"v0.7 unchanged. Review layer reversible; any triangle cut was low ink support."}
    (folder/"stats_v020.json").write_text(json.dumps(report,indent=2),
                                           encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
