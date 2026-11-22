# CityVisualizer — MapExtractor v0.5 (candidate baseline)

v0.5 promotes the **LAB-normalized** v0.4 mask to an exportable, provisional building-candidate layer. v0.2–v0.4 comparison scripts remain unchanged.

From the project root (Git CMD or PowerShell):

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\finalize.py --image maps\original\harn_sample.jpeg --config MapExtractor\sample-config.json
```

Output in `output/buildings_v05/`:
- `building_mask_v05.png`: corrected binary mask with bottom-frame exclusion;
- `building_candidates_v05.png`: candidate outlines on the original photo;
- `building_candidates_v05.geojson`: polygons in **image pixels**, NOT georeferenced;
- `building_stats_v05.json`: counts and rejection diagnostics.

The example `v05_exclusion_polygons` in `sample-config.json` masks a **narrow lower frame strip** of this specific photographed page, preventing unwanted book-border features from being exported. It is specific to the original **1536 × 1152** photograph; change/remove the coordinates for different source images. The corrected mask is used with the existing v0.2 contour selection thresholds.

The data records `classification: building_candidate`, `review_status: unreviewed`. Candidate geometries may include seating tiers, docks and other non-buildings; large structures (such as the principal temple) may be split. Tiny buildings can remain undetected. **No automatic socio-economic, historical, or building-function classification is applied.** IDs are provisional and may change in future reruns. Do not publish copyrighted original map photos.

## Previous version (v0.4 experimental comparison)

# CityVisualizer — MapExtractor v0.4 (experimental)

v0.4 adds `MapExtractor/color_compare.py` to compare the unchanged v0.2 RGB mask against a second mask made from a locally LAB-normalized photograph. It **does not replace** the existing extraction pipeline or its building IDs.

After `git pull origin main`, run from the project root in **Git CMD or PowerShell**:

```powershell
.\.venv\Scripts\python.exe MapExtractor\color_compare.py --image maps\original\harn_sample.jpeg --config MapExtractor\sample-config.json
```

The script writes `output/color_compare_v04/` with `original_mask.png`, `normalized_image.png`, `normalized_mask.png`, `mask_difference.png`, candidate overlays, cropped comparisons for `east_small_buildings` and `central_city`, and `comparison_stats.json`.

Difference map legend: **green** = newly white pixels, **magenta** = previously white pixels lost, **light gray** = white in both masks, **dark gray** = neither. These are pixel changes, **not validated building additions**. Candidate contours can split or merge; never infer buildings gained from candidate-count differences.

Optional settings in local `sample-config.json`:

```json
"v04_color_normalization": {
  "sigma_px": 95,
  "strength": 0.65,
  "max_light_shift": 24,
  "max_chroma_shift": 11
},
"v04_diagnostic_regions": {
  "east_small_buildings": [1090, 315, 1495, 550],
  "central_city": [295, 270, 620, 830]
}
```

Default settings are built into the script. Local LAB correction is exploratory, not an accurate calibration of the photographed page. It can introduce false positives in foliage or other bright areas. Review the crops before replacing any production mask.

## Previous version (v0.3)

v0.3 adds **diagnostic visualizations only**. The v0.2 extraction logic, original image mask, and filters are unchanged.

## Run diagnostics (PowerShell or Git CMD)

From the repository root, with the book photograph stored **locally** as `maps/original/harn_sample.jpeg`:

```powershell
.\.venv\Scripts\python.exe MapExtractor\diagnostics.py --image maps\original\harn_sample.jpeg --config MapExtractor\sample-config.json
```

Outputs are written to `output/diagnostics_v03/`:
- `east_small_buildings_baseline.png` and `east_small_buildings_v03.png`
- `central_city_baseline.png` and `central_city_v03.png`
- `diagnostic_stats.json` with counts per crop.

**Legend:** red = v0.1 candidate; yellow = added in v0.2; magenta = still rejected in v0.2. Crops are enlarged 3x with nearest-neighbor interpolation; this is a diagnostic overlay, not a new selection. Regions are fixed image-pixel rectangles for the supplied example and can be overridden by adding a `diagnostic_regions` object to your local config; coordinates are `[x1,y1,x2,y2]`.

**Note:** Only contours already represented in the v0.2 binary mask can be displayed as candidates or rejections. Missing mask components will require later work on segmentation. Region totals use contour bounding-box centers; they depend on the crop boundaries.

## Previous extraction (v0.2)

v0.2 changes **only the final contour selection thresholds**. The mask computation is unchanged from v0.1. This makes it easy to evaluate whether the small buildings already visible in the mask are recovered when the selection rules are relaxed.

## Setup and run (Windows PowerShell)

Extract the contents of this archive **into** `C:\Users\nattl\GIT\CityVisualizer` (not into an extra nested directory). Keep your existing `.venv` directory.

```powershell
cd C:\Users\nattl\GIT\CityVisualizer
.\.venv\Scripts\python.exe MapExtractor\extract.py --image maps\original\harn_sample.jpeg --config MapExtractor\sample-config.json
```

If upgrading from v0.1, the new `MapExtractor/extract.py` and `sample-config.json` replace those files. Existing `output/building_candidates.png` and `output/building_candidates.geojson` remain untouched.

## Outputs

- `output/candidate_mask.png` — **same pixel mask as v0.1**.
- `output/building_candidates_v02.png` — red contours = original v0.1 candidates, **yellow contours = newly included**; labels for red ones only by default to avoid clutter.
- `output/building_candidates_v02.geojson` — candidate polygons in **image pixel coordinates** (NOT WGS84 or meters). `selection` is `baseline` or `new_v02`.
- `output/selection_stats_v02.json` — counts and filter diagnostics. Rejection reason totals overlap when a shape fails more than one filter.

## Parameters

The baseline values from v0.1 remain unchanged in `baseline_filters`. The new values under `v02_filters` are less strict: minimum contour area 35→8, minimum bounding-box side 4→2, minimum fill ratio .32→.20. We do not change the image preprocessing, ROI, color thresholds, or contour finding.

The original Bxxxx IDs are preserved; added contours receive Nxxxx IDs. An ID is a **candidate identifier**, not a verified building identifier.

## Known limitations

- Small non-buildings and printing artifacts may now be included.
- Connected roofs remain merged because the segmentation mask has not changed.
- Curvature of the photographed book page is not corrected.
- No automatic identification of economic class, house function, or temples.
- This is a comparative test, not a validated extraction.
