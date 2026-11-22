# CityVisualizer — MapExtractor v0.3

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
