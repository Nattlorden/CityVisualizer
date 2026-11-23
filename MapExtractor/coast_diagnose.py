"""v0.19: diagnostics only; no changes to any coastline or water geometry.

Diagnose matched line shortcuts versus simplification, and catalog small water
bodies with a source component ID. All inputs are the original/HDR photographs.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from coast_guided import (coast_contours, normals_toward_land,
                          candidate_samples, choose_path, paths_to_lines)
from coast_refine import refine
from extract import read_image, save_png
from land_boundary_dual import ink_mask
from shoreline import v07_mask
from water import spatial_mask


def component_catalog(water, cfg):
    binary = (water > 0).astype(np.uint8)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    items = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        items.append({"id": i, "area_px": area,
                      "bbox": [x,y,width,height],
                      "centroid": [round(float(v),1) for v in centers[i]],
                      "touches_image_edge": (
                          x == 0 or y == 0 or x+width >= water.shape[1]
                          or y+height >= water.shape[0]),
                      "small": area <= int(cfg.get("small_water_area_px",2400))})
    return labels, items


def ink_support(a, b, ink, radius=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2*radius+1,2*radius+1))
    nearby = cv2.dilate(ink,kernel)>0
    n = max(2,int(np.linalg.norm(np.asarray(b)-np.asarray(a))*2)+1)
    pts = np.rint(np.linspace(a,b,n)).astype(int)
    h,w=ink.shape
    good = ((pts[:,0]>=0)&(pts[:,0]<w)&
            (pts[:,1]>=0)&(pts[:,1]<h))
    return float(np.mean(nearby[pts[good,1],pts[good,0]])) if np.any(good) else 0.


def run_diagnostics(original,hdr,profile,cfg):
    water=v07_mask(original,profile)
    allowed=spatial_mask(original.shape,profile)
    ink=ink_mask(hdr,profile["land_boundary_v012"]) & allowed
    baseline=profile["coast_guided_v015"]
    contours=coast_contours(water,allowed,baseline)
    labels,components=component_catalog(water,cfg)
    problem_links=[]
    traced=np.zeros(water.shape,np.uint8)
    count=0
    for cidx,contour in enumerate(contours):
        points=contour["points"]
        normals=normals_toward_land(points,water,baseline,contour["closed"])
        rows=candidate_samples(points,normals,ink,allowed,baseline)
        chosen=choose_path(rows,baseline)
        for j in range(1,len(chosen)):
            a,b=chosen[j-1],chosen[j]
            if a is None or b is None:
                continue
            dist=float(np.linalg.norm(np.asarray(b[1])-np.asarray(a[1])))
            if dist <= float(cfg.get("inspect_link_longer_than_px",11)):
                continue
            support=ink_support(a[1],b[1],ink)
            if support >= float(cfg.get("inspect_link_below_ink_fraction",.38)):
                continue
            problem_links.append({"contour":cidx,"from":a[1],"to":b[1],
                                  "length_px":round(dist,1),
                                  "ink_support":round(support,3)})
            cv2.line(traced,a[1],b[1],255,1)
        count+=len(paths_to_lines(chosen,baseline))
    return water,ink,traced,components,problem_links,count


def main():
    parser=argparse.ArgumentParser(description="v0.19 coastline diagnostic only")
    parser.add_argument("--image",required=True,type=Path)
    parser.add_argument("--hdr",required=True,type=Path)
    parser.add_argument("--profile",required=True,type=Path)
    parser.add_argument("--output",type=Path,
                        default=Path("output/coast_diagnostic_v019"))
    args=parser.parse_args()
    original,hdr=read_image(args.image),read_image(args.hdr)
    if original.shape!=hdr.shape:
        raise ValueError("Original and HDR must be pixel aligned")
    profile=json.loads(args.profile.read_text(encoding="utf-8"))
    cfg=profile.get("coast_diagnostic_v019",{})
    water,ink,links,components,problems,segments=run_diagnostics(
        original,hdr,profile,cfg)
    masks,lines,_,_=refine(original,hdr,profile,profile["coast_refine_v016"])
    folder=args.output
    folder.mkdir(parents=True,exist_ok=True)
    overlay=hdr.copy()
    for obj in lines:
        pts=np.rint(np.asarray(obj["line"].coords)).astype(np.int32)
        cv2.polylines(overlay,[pts],False,(0,0,220),1)
    overlay[links>0]=(225,20,225)
    water_components=hdr.copy()
    for record in components:
        if record["small"]:
            x,y,w,h=record["bbox"]
            cv2.rectangle(water_components,(x,y),(x+w,y+h),(0,215,255),1)
            cv2.putText(water_components,str(record["id"]),(x,y),
                        cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
    save_png(folder/"v016_with_unsupported_links.png",overlay)
    save_png(folder/"water_component_ids.png",water_components)
    save_png(folder/"unsupported_links.png",links)
    save_png(folder/"water_v07_unchanged.png",water)
    report={"version":"0.19","diagnostics_only":True,
            "water_components":components,
            "unsupported_match_links":problems,
            "v015_segment_count":segments,
            "v016_segment_count":len(lines),
            "warning":"Small water areas may be legitimate. No coast changes made."}
    (folder/"diagnosis_v019.json").write_text(json.dumps(report,indent=2),
                                                encoding="utf-8")
    print(json.dumps({"water_components":len(components),
                      "unsupported_links":len(problems),
                      "v016_segments":len(lines)},indent=2))


if __name__=="__main__":
    main()
