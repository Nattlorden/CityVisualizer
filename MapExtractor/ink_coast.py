"""v0.10: ink-informed water boundary. All geometries are unreviewed image pixels."""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import LineString, mapping
from extract import read_image, save_png
from water import spatial_mask, find_water_candidates
from shoreline import v07_mask

def ink_lines(image, cfg):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    k=int(cfg.get("local_window_px",23))
    if k<3 or k%2==0: raise ValueError("local_window_px must be odd and >=3")
    blurred=cv2.GaussianBlur(gray,(k,k),0)
    return (((gray.astype(np.int16)<blurred.astype(np.int16)-int(cfg.get("min_dark_contrast",19)))&
             (gray<int(cfg.get("max_ink_brightness",155)))).astype(np.uint8)*255

def color_mask(image, ranges):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask=np.zeros(hsv.shape[:2],np.uint8)
    for item in ranges:
        a=np.asarray(item["lower"],np.uint8)
        b=np.asarray(item["upper"],np.uint8)
        if a.shape!=(3,) or b.shape!=(3,):raise ValueError("Expected HSV triplets")
        mask|=cv2.inRange(hsv,a,b)
    return mask

def infer_water(image, baseline, allowed, ink, cfg):
    reach=int(cfg.get("search_distance_px",40))
    if not 5<=reach<=100:raise ValueError("search_distance_px must be 5..100")
    blue=color_mask(image,cfg.get("water_hsv_ranges",[]))&allowed
    count,labels,stats,_=cv2.connectedComponentsWithStats((blue>0).astype(np.uint8),8)
    supported=np.zeros_like(blue)
    minimum=int(cfg.get("min_blue_component_pixels",9))
    for i in range(1,count):
        if stats[i,cv2.CC_STAT_AREA]>=minimum:supported[labels==i]=255
    seed_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    water_seed=cv2.erode(baseline,seed_kernel)|cv2.erode(supported,seed_kernel)
    water_seed[ink>0]=0
    water_seed &= allowed
    search=cv2.dilate(baseline|supported,cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,(2*reach+1,2*reach+1)))
    land=color_mask(image,cfg.get("land_hsv_ranges",[]))
    near_baseline=cv2.dilate(baseline,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    land_seed=(((search==0)|((land>0)&(near_baseline==0)))&(allowed>0))
    if not np.any(water_seed) or not np.any(land_seed):
        raise ValueError("Not enough water or land seeds")
    markers=np.zeros(baseline.shape,np.int32)
    markers[land_seed]=1
    markers[water_seed>0]=2
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edge=cv2.morphologyEx(gray,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    barrier=cv2.dilate(ink,np.ones((3,3),np.uint8))
    edge=np.maximum(edge,(barrier.astype(np.float32)*float(cfg.get("ink_weight",0.8))).astype(np.uint8))
    cv2.watershed(cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR),markers)
    water=((markers==2)&(allowed>0)).astype(np.uint8)*255
    water[water_seed>0]=255
    water[(search==0)&(baseline==0)]=0
    water[allowed==0]=0
    return water,supported

def extract_shores(water, allowed, ink, cfg):
    # Pad with zeros so actual image edges are treated as clipped boundaries.
    safe=cv2.distanceTransform(np.pad((allowed>0).astype(np.uint8),1),
                               cv2.DIST_L2,3)[1:-1,1:-1]
    contours,hierarchy=cv2.findContours(water,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:return []
    features=[]
    ink_near=cv2.dilate(ink,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
    for idx,contour in enumerate(contours):
        pts=contour.reshape(-1,2)
        good=safe[pts[:,1],pts[:,0]]>float(cfg.get("map_edge_clearance_px",3))
        if not np.any(good):continue
        if np.all(good):
            runs=[np.vstack((pts,pts[0]))]
        else:
            cut=int(np.flatnonzero(~good)[0])
            pts=np.roll(pts,-cut,axis=0)
            good=np.roll(good,-cut)
            runs=[];start=None
            for j,isgood in enumerate(good):
                if isgood and start is None:start=j
                if not isgood and start is not None:runs.append(pts[start:j]);start=None
            if start is not None:runs.append(pts[start:])
        for run in runs:
            if len(run)<2:continue
            line=LineString(run.tolist()).simplify(float(cfg.get("simplify_px",0.8)),
                                                   preserve_topology=True)
            if line.length<float(cfg.get("min_shore_length_px",7)):continue
            subsample=run[::max(1,len(run)//200)]
            evidence=float(np.mean(ink_near[subsample[:,1],subsample[:,0]]))
            fid=f"SH{len(features)+1:04d}"
            features.append({"type":"Feature","id":fid,"properties":{
                "id":fid,"layer":"shoreline","classification":"shoreline_candidate",
                "review_status":"unreviewed","ink_support_fraction":round(evidence,3),
                "boundary_role":"water_hole" if hierarchy[0][idx][3]>=0 else "water_exterior",
                "coordinate_system":"image_pixels_top_left_origin"},
                "geometry":mapping(line)})
    return features

def main():
    p=argparse.ArgumentParser(description="Experimental ink-informed shoreline v0.10")
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--profile",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("output/ink_coast_v010"))
    args=p.parse_args()
    image=read_image(args.image)
    profile=json.loads(args.profile.read_text(encoding="utf-8"))
    opts=profile.get("ink_boundary",{})
    allowed=spatial_mask(image.shape,profile)
    old=v07_mask(image,profile)
    ink=ink_lines(image,opts)
    water,blue=infer_water(image,old,allowed,ink,opts)
    shores=extract_shores(water,allowed,ink,opts)
    polygons,_=find_water_candidates(water,profile)
    water_features=[]
    for i,(_,area,geom,_) in enumerate(polygons,1):
        fid=f"W{i:04d}"
        water_features.append({"type":"Feature","id":fid,"properties":{
            "id":fid,"classification":"water_candidate","review_status":"unreviewed",
            "area_px2":round(area,1),"coordinate_system":"image_pixels_top_left_origin"},
            "geometry":mapping(geom)})
    args.output.mkdir(parents=True,exist_ok=True)
    added=(water>0)&(old==0)
    removed=(old>0)&(water==0)
    overlay=image.copy()
    overlay[added]=(30,220,30)
    overlay[removed]=(220,40,180)
    overlay=cv2.addWeighted(image,0.55,overlay,0.45,0)
    for item in shores:
        points=np.asarray(item["geometry"]["coordinates"],np.int32)
        if len(points)>=2:
            c=(0,220,255) if item["properties"]["ink_support_fraction"]>=0.35 else (0,0,230)
            cv2.polylines(overlay,[points],False,c,1)
    for name,mask in (("water_mask",water),("v07_baseline",old),
                      ("ink_candidates",ink),("blue_candidates",blue)):
        save_png(args.output/f"{name}_v010.png",mask)
    save_png(args.output/"ink_coast_overlay_v010.png",overlay)
    for name,features in (("water",water_features),("shoreline",shores),
                          ("littoral_zone",[]),("coastal_terrain",[])):
        (args.output/f"{name}_v010.geojson").write_text(json.dumps({
            "type":"FeatureCollection","features":features},indent=2),encoding="utf-8")
    stats={"version":"0.10","experimental":True,
           "v07_water_pixels":int(np.count_nonzero(old)),
           "v010_water_pixels":int(np.count_nonzero(water)),
           "new_water_pixels":int(np.count_nonzero(added)),
           "removed_water_pixels":int(np.count_nonzero(removed)),
           "shoreline_segments":len(shores),"water_polygons":len(water_features),
           "ink_supported_shorelines":sum(x["properties"]["ink_support_fraction"]>=0.35 for x in shores),
           "note":"Ink is a boundary cue, not a verified coast. Littoral/material layers intentionally empty."}
    (args.output/"stats_v010.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    print(json.dumps(stats,indent=2))
    print(f"Saved to {args.output.resolve()}")

if __name__=="__main__":
    main()
