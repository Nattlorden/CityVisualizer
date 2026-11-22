# CityVisualizer / MapExtractor v0.1

First proof of concept: color thresholding on a photographed Hârn-style city map.

## Requirements
Python 3.14, NumPy, OpenCV, Pillow, Shapely.

```powershell
cd C:\Users\nattl\GIT\CityVisualizer
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe MapExtractor\extract.py --image maps\original\harn_sample.jpeg --config MapExtractor\sample-config.json
```

Result under `output/`:
- `candidate_mask.png`: thresholded pixels.
- `building_candidates.png`: numbered proposed contours over the original.
- `building_candidates.geojson`: approximate polygons in **pixel coordinates**, NOT geographic WGS84 coordinates or meters.

## Limitations
- This is a proof of concept, **not** a reliable automated building recognizer.
- Many individual attached houses can be merged and some non-buildings falsely detected.
- The sample ROI is tuned to the supplied photograph; change `roi_polygon` for other maps, or omit it for the whole image.
- Perspective, page curvature, labels, and illumination are not corrected yet.
- Do not treat extracted IDs as definitive building IDs until reviewed.
- GeoJSON is used as a polygon container in image coordinates, without CRS.

Next: sample map rectification, interactive polygon correction, and better roof segmentation.
