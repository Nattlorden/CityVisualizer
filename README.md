# CityVisualizer — MapExtractor v0.9 (four coastal layers)

v0.9 uses **v0.7 as a stable water baseline** and proposes a conservative land-aware extension. All thresholds, map boundaries and example seed points are profile data; the engine has no city names or fixed geometry. **The four layers are provisional candidate data**, not validated coastlines or seabed information.

From the repository root (Git CMD or PowerShell):

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\coastal.py --image maps\original\harn_sample.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Inspect `output/coastal_v09/coastal_overlay_v09.png` first. Color legend: **blue tint** = detected water; **yellow-green tint** = water-side proximity band; **orange tint** = land-side coastal terrain band; **red line** = candidate shoreline. The additional `added_water_v09.png` shows changes from v0.7 alone.

Four GeoJSON exports (all in **image pixel** coordinates):

- `water_v09.geojson`: water candidates including local color-guided expansion from v0.7;
- `shoreline_v09.geojson`: LineStrings tracing candidate land/water boundaries, deliberately omitting image and profile clip edges;
- `littoral_zone_v09.geojson`: provisional water-side proximity zone, **not evidence of shallowness**;
- `coastal_terrain_v09.geojson`: provisional land-side proximity zone. Material is `unknown` until later classification or manual review.

Additional mask PNGs and `coastal_stats_v09.json` are written for diagnostics. Optional settings are under `coastal_layers` in the water profile. `boundary_growth.max_distance_px` bounds expansion (default in sample: 12); `water_hsv_ranges` and `land_hsv_ranges` are configurable evidence filters. `littoral_band_px` and `coastal_land_band_px` describe image-space bands, **not real-world distances**. Profiles must be recalibrated when using different source maps.

Test without the proprietary image:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s MapExtractor -p "test_coastal.py"
```

**Important limitations:** Shoreline geometry is extracted from the refined mask; it is not yet a reliable vectorization of the cartographer's ink boundary. Missing narrow waterways may remain missing, and island/shore classification remains provisional. Zones are geometry placeholders for future lithology, sand, stones, mixed beaches, reeds and quay segmentation. Do not infer real bathymetry or material from these bands. The reference map photo must stay local (`maps/original/` is ignored by Git).

## Previous version (v0.8 experimental shoreline)

# CityVisualizer — MapExtractor v0.8 (experimental shoreline)

This iteration **does not replace v0.7**. It treats the v0.7 mask as initial water, then uses OpenCV GrabCut inside a narrow editable boundary band to propose a revised shoreline. Image edges are shown separately as diagnostic evidence, not blindly followed (printed labels and piers can also produce dark lines).

From the repository root, using Git CMD or PowerShell:

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\shoreline.py --image maps\original\harn_sample.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Look at `output/shoreline_v08/shoreline_overlay_v08.png`: **orange** = experimental shoreline, **green** = water added versus v0.7, **magenta** = water removed. `shoreline_dark_line_evidence_v08.png` contains Canny lines found near the old boundary; these are **unclassified visual cues**, not shoreline polygons.

Other outputs: `water_mask_v07_baseline.png`, `water_mask_v08_experimental.png`, `shoreline_edge_v08.png`, `water_candidates_v08.geojson`, `shoreline_candidates_v08.geojson` (LineStrings including island shores), and `shoreline_stats_v08.json`. GeoJSON coordinates remain **image pixels**, not ground meters. All candidates are unreviewed, and IDs remain provisional.

Map-specific tuning lives in the profile's `shoreline` section: `band_px` (default 5; how far the boundary may move), `iterations` (default 3), Canny thresholds, minimum shoreline length, and line simplification. The generic algorithm contains no city coordinates, named water bodies or fixed color palette. This is an experiment: GrabCut can still follow lettering or damage narrow canals, and the polygon and coastline layers have not been manually verified. Always compare against v0.7 before adoption.

## Previous version (v0.7 water refinement)

# CityVisualizer — MapExtractor v0.7 (water refinement)

v0.7 **retains the v0.6 water mask** and adds optional refinement stages, configured through the map-specific water profile (not hardcoded in the extractor):

1. `shore_refinement`: detect pixels matching a wider HSV range **only within `distance_px` of existing water**.
2. `seeded_water`: seeded flood-fill in LAB around user-selected points, constrained by a local radius and the profile's map exclusions.

From the project directory, after `git pull origin main`:

```powershell
.\.venv\Scripts\python.exe MapExtractor\water_refine.py --image maps\original\harn_sample.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Output in `output/water_v07/`: `water_mask_v07.png`, `water_overlay_v07.png`, `water_new_shore_v07.png`, `water_seeded_v07.png`, `water_candidates_v07.geojson`, `water_stats_v07.json`.

Overlay legend: **blue** original v0.6 water, **green** added shoreline, **orange** seeded areas. Boundaries are outlined in blue. If the added shoreline includes land, reduce `shore_refinement.distance_px` (currently 7) or tighten its HSV range. If seeded water includes marshland, lower the `lab_tolerance` array or reduce `max_radius_px`. `seeded_water.points` are **example pixel coordinates for this photograph only**. Set `points: []` when processing an unrelated map. The profile is a starting point and must be visually reviewed.

Run the synthetic, map-independent smoke test:

```powershell
.\.venv\Scripts\python.exe MapExtractor\test_water_refine.py
```

**Limitations:** v0.7 is still provisional; narrow waterways may remain disconnected and some shorelines may be offset. Seed points are a temporary route to mapping isolated ponds; they are not an automatic monochrome map solution. Another map can reuse the engine with a newly prepared profile, while B/W maps will require another detector. No original copyrighted source map is included in the public repository.

## Previous version (v0.6 water candidates)

# CityVisualizer — MapExtractor v0.6 (water candidates)

The **water layer** is independent from the v0.5 building layer. Detection logic lives in `MapExtractor/water.py`; **map-specific HSV thresholds and spatial limits** live in replaceable JSON profiles. Example profile: `MapExtractor/profiles/harn_color_photo.water.json`.

From the project root after `git pull origin main`:

```powershell
.\.venv\Scripts\python.exe MapExtractor\water.py --image maps\original\harn_sample.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Outputs in `output/water_v06/`:
- `water_mask_v06.png`: binary color-based water mask;
- `water_candidates_v06.png`: provisional blue outlines on original;
- `water_candidates_v06.geojson`: polygons, potentially with holes, in **image pixel coordinates**;
- `water_stats_v06.json`: simple diagnostics.

**Important:** All water objects are `water_candidate` and `unreviewed`; the method may mistake roofs, shadows, vegetation or printed text for water and may fragment canals. This is a first-pass, *not a validated coastline*. The provided color profile is calibrated only approximately for our test photograph; different maps need different profiles. `include_polygons` and `exclude_polygons` refer to that image's pixel coordinates and are optional. Remove them or redraw them for any other image, especially scanned pages with different dimensions. The engine has no fixed city-specific coordinates.

**Black-and-white maps:** HSV color detection does not apply. A future method will combine linework, textures, region connectivity and user seed points, while retaining the same GeoJSON water-layer output. We have deliberately not imposed a blue-color assumption on the data model.

No original Hârn map image is committed to this public repository. Do not commit `maps/original/`.

## Previous version (v0.5 buildings)

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
