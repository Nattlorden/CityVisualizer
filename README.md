# CityVisualizer v0.21 — Fortifications (first diagnostic pass)

The coastline work is paused: **v0.16 `hdr_candidate_comparison` stays the accepted coast baseline**. New work begins on distinct fortification classes:

1. `wall_segment_candidate` — elongated or thick continuous dark masonry, not yet verified as walls.
2. `tower_or_bastion_candidate` — locally thicker dark structure, which may turn out to be towers, gates, buildings, or ordinary wall junctions.
3. `possible_wall_gap_not_verified_gate` — small light interruptions between dark stretches. **A gap is not proof of a gate.**
4. `possible_fortification_building` — a large roof-colored building candidate close to wall-like masonry. Size and adjacency alone do not establish military purpose.

Use the HDR photo to emphasize dark walls; do **not** run on the previously generated coast overlay (which contains artificial black/red/blue marks).

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\fortifications.py --image maps\original\harn_sample_hdr.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json --building-config MapExtractor\sample-config.json
```

Inspect `output/fortifications_v021/fortification_candidates_overlay.png`. Color coding is **blue** wall candidates, **magenta** potential towers/thickenings, **yellow** possible gate gaps, and **green** large possible fortification buildings. Separate PNG diagnostic masks and GeoJSON candidate polygons are written for every type. `stats_v021.json` contains preliminary counts. Geometries are in **image pixel coordinates**.

**Very important:** The v0.21 detector is intentionally broad. Dark roads, map text, urban house outlines and vegetation can still be selected. The first run is to assess the dark masonry mask, **not** an accurate fortified-city segmentation. The profile's `fortifications_v021.regions` can optionally restrict analysis to selected fortification neighborhoods, using image pixel polygons. The code contains no map-specific coordinates. Tower, gate, or fortification-building labels are unreviewed suggestions.

Initial synthetic checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s MapExtractor -p "test_fortifications.py"
```

The source photograph has not yet been used to validate those semantic labels. Next steps after inspecting the overlay are to identify a few reliably fortified wall segments as structural seeds, trace the connected defensive perimeter, and only then propose attached towers, gates, and building sections. Keep fortification buildings as polygons and walls as separate connected geometry, not as a single undifferentiated dark mask.

## Earlier coastal checkpoint

# CityVisualizer — coastal checkpoint after v0.20

**Decision point:** v0.16 `hdr_candidate_comparison.png` is the best *user-verified* baseline. v0.17 and v0.18 did not demonstrate meaningful improvements. Two final, experimental attempts were added: v0.19 diagnostics and v0.20 targeted, reversible selection. Neither replaces v0.16 unless visual comparison shows a clear improvement. This is a deliberate pause point for coastal extraction.

## v0.19 — non-destructive diagnostics

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\coast_diagnose.py --image maps\original\harn_sample.jpeg --hdr maps\original\harn_sample_hdr.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Review `output/coast_diagnostic_v019/water_component_ids.png`: independent small water shapes are boxed and identified. `v016_with_unsupported_links.png` shows weak-ink links in magenta over the unchanged red v0.16 candidate. `diagnosis_v019.json` records water component areas and possible unsupported match connections.

## v0.20 — final conservative candidate experiment

```powershell
.\.venv\Scripts\python.exe MapExtractor\coast_checkpoint.py --image maps\original\harn_sample.jpeg --hdr maps\original\harn_sample_hdr.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Review these side-by-side in `output/coast_checkpoint_v020/`:

- `hdr_reference_v016.png`: red lines from v0.16, unchanged
- `hdr_candidate_v020.png`: proposed red coastal candidates **without review-only small-water lines**, and with only strongly unsupported inland triangle tips omitted
- `hdr_review_v020.png`: same proposed red geometry plus **yellow** review-only lines near independent little water components
- `small_water_components.png`: isolated small water bodies used as evidence, not a land classification
- `coast_candidates_v020.geojson`: surviving lines
- `review_small_water_v020.geojson`: excluded candidates retained for recovery
- `triangle_changes_v020.json`: original source IDs and deleted-tip coordinates
- `water_v07_unchanged.png`: original water pixels, always preserved

The appearance of extra yellow is **not an improvement**; it only makes uncertainty visible. Success requires fewer false triangles and missing Pac-Man candidates in the main red layer **without losing correct natural coastline, islands or sharp quays**. If the side-by-side review fails this standard, **retain v0.16**.

The user-supplied photographs remain source material; do not add new copies to the repository. Existing Git history may already contain one user-pushed HDR photograph.

### Validation and future work

```powershell
.\.venv\Scripts\python.exe -m compileall -q MapExtractor
.\.venv\Scripts\python.exe -m unittest discover -s MapExtractor -p "test_coast_*.py"
```

Synthetic tests were added for water component labeling, ink support, small water review, triangle trimming and preserved ink-supported quays. Results on the real photographed map have **not** yet been independently verified; inspect the output before accepting any new baseline.

When resuming: review the small marsh water component IDs and any incorrectly tagged island, study why v0.7 misses east-side bays, add manual control points only where genuinely needed, and avoid general image-color threshold retuning that previously destroyed good coastlines.

## Previous development log

# CityVisualizer v0.18 — Matching geometry, not post-cleanup (experimental)

Retains **v0.16 `hdr_candidate_comparison` as the visual reference**. This version adjusts the candidate-selection stage itself, while leaving v0.7 water, v0.15, v0.16 and v0.17 untouched.

- The dynamic-programming path chooser now tests the **actual distance between successive HDR ink intersections**. A sufficiently long step with no matching dark ink along its connecting line is invalid, allowing a gap instead of a diagonal "triangle". Real, sharp quay corners and genuinely ink-backed long segments are retained.
- Small, closed, fairly rounded **exterior water contours** are treated as separate *review-only* inputs rather than forcing an inner coastal boundary around little blue ponds. Closed **holes inside water** (which can represent islands) and open map boundaries are **not** excluded by this heuristic.
- The original v0.7 water mask is never modified; quarantined traces are exported separately and recoverable.

Run (repo root, PowerShell or Git CMD):

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\coast_matching.py --image maps\original\harn_sample.jpeg --hdr maps\original\harn_sample_hdr.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Inspect `output/coast_matching_v018/`:

- `hdr_reference_v016.png`: the winning red geometry (recomputed from v0.16).
- `hdr_matching_v018.png`: proposed v0.18 geometry. **Red** = ordinary matching candidate; **yellow** = small-water tracing in the separate review layer; **blue** = unchanged v0.7 water edge.
- `matched_hits_v018.png` and `small_water_review_hits_v018.png`: accepted matches and small-water matches, separately.
- `water_v07_unchanged.png`: untouched water mask.
- `coast_candidates_v018.geojson`: candidate inner land boundary lines.
- `small_water_review_v018.geojson`: retained possible Pac-Man/pond traces for inspection.
- `stats_v018.json`: skipped transitions, unmatched samples and segment counts.

Settings under `coast_matching_v018` in the existing profile:
`max_point_step_px` (35), `verify_step_from_px` (10), `min_ink_link_fraction` (0.28), `small_water_max_area_px2` (2200) and `small_water_min_circularity` (0.32). These are preliminary, deliberately tunable values; they are not verified to fix every triangle in the actual photographed map. Contours not represented in the original v0.7 mask (notably missing bays) cannot be recovered by this step.

Synthetic tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s MapExtractor -p "test_coast_matching.py"
```

The original book-map photographs should not be redistributed as part of this public project's test assets. The public repository currently contains a user-committed HDR photograph, which is a separate copyright concern.

## Previous version (v0.17)

# CityVisualizer v0.17 — Conservative geometry audit (experimental)

v0.16's **`hdr_candidate_comparison` is the reference**. This version does **not** change the v0.7 water surface, HDR ink extraction, coast matching, or v0.16 code. It adds a reversible review/cleanup step.

- Long, unsupported straight connectors between v0.16 vertices are split **only if** the connecting pixels have little matching ink in the original HDR mask. A long quay with a genuinely dark straight edge remains untouched.
- Very sharp, short triangle tips are removed **only** when they also lack supporting HDR ink. Genuine sharply drawn headlands and quays are not automatically smoothed.
- Small, round v0.7 water components are flagged as **possible causes** of the dotted Pac-Man arcs. Nearby short inner-boundary proposals move into a **review-only GeoJSON layer**, but the lake/water pixels and source v0.16 traces remain unchanged.

Run from repository root (Windows PowerShell or Git CMD):

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\coast_geometry.py --image maps\original\harn_sample.jpeg --hdr maps\original\harn_sample_hdr.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Review `output/coast_geometry_v017/`:

- `hdr_reference_v016.png`: the earlier winning red candidate on HDR, reproduced from v0.16 geometries (without classifications)
- `hdr_geometry_v017.png`: proposed result: **red** retained candidates, **yellow** small-round-water review candidates, **magenta** flagged unsupported lines or triangle tips, **blue** unchanged v0.7 water edge
- `geometry_changes_v017.png`: grayscale mask of unsupported shortcuts/tips that v0.17 identified
- `suspect_round_water_v017.png`: small round *v0.7* outlines which deserve inspection, **not automatically erroneous lakes**
- `water_v07_unchanged.png`: untouched baseline
- `coast_candidates_v017.geojson`: retained candidate coastline parts
- `coast_review_v017.geojson`: quarantined suspect Pac-Man traces, available for restoration
- `geometry_changes_v017.json`: reason and coordinates of each flagged geometric change
- `stats_v017.json`: counts and retained version metadata

Settings are in `coast_geometry_v017` in the water profile. `max_unsupported_link_px` defaults to 15; `min_link_ink_fraction` defaults to 0.28; `round_review_radius_px` defaults to 20. All are first-pass conservative values requiring visual validation. If an authentic coastline is flagged, keep v0.16 and adjust parameters rather than deleting the segment.

Run syntax check and the synthetic regression suite:

```powershell
.\.venv\Scripts\python.exe -m compileall -q MapExtractor
.\.venv\Scripts\python.exe -m unittest discover -s MapExtractor -p "test_*.py"
```

**Known limitations:** The small-loop detector infers shape from v0.7; a round water feature may be a legitimate small lake or island. Geometry splitting after simplification cannot repair a fundamentally incorrect choice of dark line. No automatic bay correction or new littoral polygon is claimed. Results are unreviewed.

## Previous version (v0.16)

# CityVisualizer v0.16 — Conservative cleanup and coastal context (experimental)

**v0.15 is preserved and unchanged.** The best result so far is the red coastline on the HDR overlay. This version deliberately keeps that matching algorithm and applies a separate post-processing stage.

From the repository root:

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\coast_refine.py --image maps\original\harn_sample.jpeg --hdr maps\original\harn_sample_hdr.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Outputs in `output/coast_refined_v016/`:

- `hdr_refined_overlay.png`: **green** = candidate natural coast, **orange/blue** = candidate built quay/coast, **red** = segments requiring review. **Blue** thin outer line = unchanged v0.7 reference.
- `original_refined_overlay.png`: same candidates on the original photograph.
- `hdr_candidate_comparison.png`: same candidate geometry in red for direct comparison against v0.15.
- `review_required.png`: grayscale mask of uncertain and short candidates.
- `water_v07_unchanged.png`: proof that v0.7 remains an untouched outer reference.
- `coast_candidates_v016.geojson`: pixel-coordinate LineStrings with `type`, `confidence`, `review_status`, and review reasons including possible quay spurs.
- `stats_v016.json`: counts per provisional terrain class.

v0.16 does **not** claim to fill missing bays. Where v0.7 misses a bay or its normal points the wrong way, the inner edge may still be wrong or absent; suspicious twists and short runs are flagged for review. Short, clearly backtracking spikes can be pruned without flattening legitimate headlands. Natural versus built coast is a *context-dependent candidate label*, not a verified shoreline type. Water and littoral polygons are unchanged; no new coastland polygon is inferred.

Set thresholds under `coast_refine_v016` in the existing water profile. Important controls: `spike_span_px` (maximum short artifact width), `spike_detour_ratio`, `min_reliable_length_px`, `built_review_length_px`, `winding_review_threshold`, and HSV ranges for natural/built surroundings.

Tests (synthetic only):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s MapExtractor -p "test_coast_refine.py"
```

The first evaluation should compare `hdr_refined_overlay.png` with the **successful v0.15** `hdr_overlay.png` near the southern wooded shore and harbor, rather than assuming an improved result. This version is still unreviewed.

## Previous version (v0.15)

# CityVisualizer v0.15 — Water-guided HDR ink matching (experimental)

This approach uses the v0.7 water-mask boundary as the **starting reference** for finding the separate, darker **inner land/shore boundary** in the HDR image. It does **not** alter or relabel the v0.7 water boundary.

For every separate water contour (including hole boundaries around islands), it samples points and determines the normal pointing away from water. It searches along each normal for HDR dark-ink intersections. Dynamic programming chooses a relatively continuous series of distances, preferring nearby ink without permitting large sudden jumps. Where no plausible ink exists, it emits a **gap**, not an invented coast segment. Separate contours stay separate.

Run from repo root:

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\coast_guided.py --image maps\original\harn_sample.jpeg --hdr maps\original\harn_sample_hdr.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Inspect `output/coast_guided_v015/`:

- `original_overlay.png`, `hdr_overlay.png`: **blue** = outer v0.7 water edge, **yellow** = individual accepted HDR-ink intersection hits, **red** = provisional joined inner-shore segments.
- `selected_ink_hits.png`: chosen ink hits without background.
- `sampled_rays.png`: sparse sampling normals (for debugging, not coast geometry).
- `ink_hdr_raw.png`: unchanged HDR ink detector input; can still include all text, houses and trees.
- `water_v07.png`: baseline reference.
- `land_boundary_v015.geojson`: unreviewed inner boundary LineStrings in original image pixel coordinates.
- `stats_v015.json`: candidate counts and missing matched samples.

Tune `coast_guided_v015` in `MapExtractor/profiles/harn_color_photo.water.json`: `min_offset_px` skips the water-edge ink; `max_offset_px` defines the maximum possible beach width; `sample_step_px` controls contour sampling; `distance_jump_penalty` and `max_step_offset_change_px` control continuity; `gap_penalty` and `ink_reward` control the decision to accept ink versus leave an unknown gap. All settings are initial guesses, **not calibrated** on this actual Hârn image.

Optional: `--water-mask path/to/a/saved-v07-mask.png` uses an existing mask in the source image's pixel dimensions.

Regression tests (synthetic map with a true dark coast and a farther building; and an actual gap):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s MapExtractor -p "test_coast_guided.py"
```

**Limitations:** Proximity alone does not identify semantic geography. A house or quay may still win if its line is closer and similarly continuous. The first detected dark line is not guaranteed to be the green-land/shore contour; curved/narrow banks can make normal direction ambiguous. Only visually validated results should be used for downstream geometry. This experiment creates no enclosed beach polygons and requires no manual seed clicks.

## Previous versions

# CityVisualizer v0.14 – Two-sided coastal ink selection (experimental)

The feedback from v0.13 was that `ink_hdr` detects the real coast *and also* houses, trees and lettering. v0.14 **does not change ink detection**. Instead it tests the colors on **opposite sides of each ink pixel**, following the broad local direction of green-land color change. An ink pixel is selected if it separates green land from ochre/sand shore, and lies within a configurable maximum distance of v0.7 water.

This is a **stricter filter**, not a final coastline: it may omit shaded cliffs, quays, rocky beaches or interrupted ink. Keep all diagnostic layers rather than assuming the selection is correct.

Run from the repo root:

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\ink_coast_select.py --image maps\original\harn_sample.jpeg --hdr maps\original\harn_sample_hdr.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Files appear in `output/ink_coast_v014/`:

- `ink_hdr_raw.png`: **unaltered** dark-line detector.
- `green_color.png`, `shore_color.png`: evidence for opposing sides.
- `side_score.png`, `side_separation.png`: diagnostic confidence, **not** final coast probabilities.
- `side_evidence.png`: ink pixels with qualifying opposite-side color contrast.
- `side_skeleton.png`: thinned selected ink.
- `bridges.png`: separately stored suggested short links.
- `connected.png`: selected skeleton plus links.
- `land_boundary_candidates.png`: filtered, connected candidate pixels.
- `selection_vs_all_ink.png`: gray = discarded dark ink, red = selected candidate, magenta = unverified gap connection.
- `hdr_coast_overlay.png` and `original_coast_overlay.png`: same candidates on both original maps.
- `land_boundary_candidates_v014.geojson`: unreviewed LineStrings; no filled polygons.
- `stats_v014.json`: counts for raw ink, two-sided qualifying ink, links and final selected pixels.

Tune `ink_coast_v014` in `MapExtractor/profiles/harn_color_photo.water.json`. Most relevant options are `side_offset_px` (11), `color_smoothing_sigma_px` (5), `normal_sigma_px` (11), `min_side_color_fraction` (0.22), `min_side_separation` (0.16), and `max_water_distance_px` (150). To reduce false positives increase `min_side_separation` or `min_side_color_fraction`; to recover a missing valid coast decrease them. Color-range adjustments can affect both.

**Important:** This method focuses on the **green-land / sandy-shore boundary**, not every possible water/land boundary. City edges, quays, completely blue lakes and bare stone need different material-specific rules. Islands are not excluded: each land/shoreside contrast may produce independent segments. v0.7 remains the unchanged outer water reference.

Run the synthetic regression test:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s MapExtractor -p "test_ink_coast_select.py"
```

The thresholds have **not** yet been proven on the real photographed map. Review `selection_vs_all_ink.png` first to see what is lost, not merely what remains.

## Previous version (v0.13)

# CityVisualizer — MapExtractor v0.13: HDR ink connectivity (experimental)

This iteration uses the HDR dark-ink mask **directly**, rather than extracting boundaries of green land regions. It thins linework and proposes connections only between nearby, facing endpoints of different components. Multiple separate coastline segments and islands are allowed. v0.7 remains an unchanged water reference.

Run in Git CMD or PowerShell from the repository root:

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\ink_connect.py --image maps\original\harn_sample.jpeg --hdr maps\original\harn_sample_hdr.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Inspect `output/ink_connect_v013/`:

- `ink_hdr_raw.png`: the original HDR ink detection.
- `ink_hdr_skeleton.png`: thinned linework before connections.
- `bridges.png`: new, **unreviewed** gap connections.
- `ink_hdr_connected.png`: all thinned ink plus bridges.
- `all_connected_overlay.png`: all connected ink on the HDR image.
- `coastal_corridor.png`: broad shore/water proximity evidence.
- `coastal_selection.png` and `coastal_candidates_overlay.png`: separately filtered candidates. Comparing this to the all-connected version reveals when selection discards useful coastline.
- `land_boundary_candidates_v013.geojson`: ordered graph LineStrings (image pixel coordinates).
- `bridge_proposals_v013.json`, `stats_v013.json`: traceability.

**Overlay legend:** yellow/orange = detected ink lines; magenta = added gap connections. None are yet confirmed coastal boundaries.

Tune settings in `ink_connect_v013` inside the sample water profile: `max_gap_px` (12), `min_direction_cos` (0.45, stricter means fewer bridges), `max_water_distance_px` (110), and `min_corridor_fraction` (0.2). This is not calibrated to the real HDR map yet; the engine contains no Golotha-specific coordinates.

Synthetic tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s MapExtractor -p "test_ink_connect.py"
```

**Limitations:** Crossing lines, longer gaps, map labels and vegetation can still cause false or missed links. The result is a diagnostic proposal, not a verified continuous coast or littoral polygon. v0.7 water is not overwritten. The repository contains a user-committed HDR copy of a book map; avoid redistributing it without appropriate permission.

## Previous version (v0.12)

# CityVisualizer v0.12 — Dual-image land/strand tracing (experimental)

This version compares the original photograph with the user-enhanced HDR photograph and proposes **multiple separate** land-to-beach boundaries: mainland segments, separate islands, and any other sufficiently supported green land regions. It deliberately does NOT assume a single coastline and does NOT modify v0.7 water or synthesize shoreline-zone polygons.

From the repository root, after `git pull origin main`:

```powershell
.\.venv\Scripts\python.exe MapExtractor\land_boundary_dual.py --image maps\original\harn_sample.jpeg --hdr maps\original\harn_sample_hdr.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Compare `output/land_boundary_v012/original_overlay.png` and `hdr_overlay.png`. Both use **exactly the same extracted boundary candidates**: yellow indicates greater support from nearby dark ink (at least 0.35), red less support. `land_boundary_v012.geojson` stores separate unreviewed LineStrings, source-component IDs and whether an extracted contour segment forms a closed ring. Diagnostic masks include `green_original.png`, `green_hdr.png`, `green_consensus.png`, `sand_evidence.png`, `ink_original.png`, `ink_hdr.png`, `land_regions.png`, and `water_v07_reference.png`. The count of separate source components and traces appears in `stats_v012.json`.

`MapExtractor/profiles/harn_color_photo.water.json` contains image-specific color thresholds and a bounded search radius. The implementation accepts any pair of aligned images with the same dimensions; a different map needs its own profile. The current profile is a first approximation, **not calibrated** or verified with the new HDR photograph. HDR contrast can increase unwanted lines from vegetation and buildings.

**Critical limitations:** These are contours of *inferred green areas*, not guaranteed traces of the original black ink. They may be shifted from ink or interrupted by trees, and closed loops can include false positives. The reported island components are *candidate regions*, not confirmed island classifications. Map-edge artifacts are suppressed rather than described as actual coast. The image must remain in its original pixel coordinate system; v0.7 still determines the outer water transition. We must visually evaluate these overlays before using any results as real 3D geometry.

All regression tests can be run locally with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s MapExtractor -p "test_*.py"
```

## Previous version (v0.11)

# CityVisualizer — MapExtractor v0.11: separate inner land boundary (experimental)

v0.11 adds `MapExtractor/land_boundary.py`. **Keep v0.7 as the outer shore-zone / water transition**, and separately detect **dark ink at the green-land / sand-colored shoreline transition** as candidate inner boundaries. This does **not** modify water, rebuild the coast, or generate a closed littoral polygon.

From the repository root, in Git CMD or PowerShell:

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\land_boundary.py --image maps\original\harn_sample.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Review `output/land_boundary_v011/inner_vs_outer_v011.png`: **blue** = unchanged outer water edge (v0.7); **red** = independent, *unreviewed* candidate inner land contours from dark ink. See `inner_boundary_candidates.png` and `land_edge_evidence.png` to understand the detection. `land_boundary_v011.geojson` contains fragments as LineStrings in **image pixel coordinates**. `stats_v011.json` counts fragments, not verified shorelines.

The sample profile's `land_boundary_v011` section contains the adjustable green-land, sand-colored zone and ink thresholds. All source-image-specific settings belong in the JSON, **not in the extractor**. **Known limitations:** fragments are disconnected; many buildings, trees and other printed marks may still be proposed. Very pale or uninked shore portions are omitted. This is an *ink candidate diagnostic*, **not an accurate continuous tracing of the cartographer's line**. In particular we do not infer a filled polygon between inner and outer boundaries until those lines have been validated and connected. A true ink-following graph tracer / manual editing tool is a later step.

Optional `--water-mask output/water_v07/water_mask_v07.png` uses an already saved v0.7 mask instead of recomputing it; the mask must match the original photo dimensions.

Synthetic check:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s MapExtractor -p "test_land_boundary.py"
```

Only a local exploratory test on the original photograph with a **v0.6 approximation** of the outer water mask has been completed before publication, not a complete v0.7-integrated verification. Compare the resulting red fragments against the dark land contour, especially the two previously highlighted regions. Do not treat the exported fragments as finalized shoreline.

## Previous version (v0.10 ink-informed water)

# CityVisualizer — MapExtractor v0.10: ink-informed shoreline (experimental)

This version tests **dark map linework as a shoreline boundary cue**, using color evidence to seed water versus land. It builds on the v0.7 water baseline, but unlike v0.9 it allows region classification beyond the previous narrow shore band. The method remains probabilistic: dark strokes can be text, buildings, piers and marsh symbols.

**Run from the repo root** (Git CMD or PowerShell):

```powershell
git pull origin main
.\.venv\Scripts\python.exe MapExtractor\ink_coast.py --image maps\original\harn_sample.jpeg --profile MapExtractor\profiles\harn_color_photo.water.json
```

Inspect `output/ink_coast_v010/ink_coast_overlay_v010.png`. **Green** = proposed new water compared with v0.7; **magenta** = water removed; **yellow** = shoreline candidate with stronger nearby ink evidence; **red** = lower ink support. The dark-ink diagnostic is `ink_candidates_v010.png`. Blue-candidate seeds are shown in `blue_candidates_v010.png`. Other outputs include `water_mask_v010.png`, `v07_baseline_v010.png`, `water_v010.geojson`, `shoreline_v010.geojson`, and `stats_v010.json`.

`littoral_zone_v010.geojson` and `coastal_terrain_v010.geojson` intentionally contain **no fabricated geometries**. A visually lighter band is not by itself proof of shallow water; we will add separate evidence-based segmentation later. Beach materials (sand, stones, mixed) are also **not inferred** in this version.

The example profile's `ink_boundary` section controls local darkness, seed color ranges, maximum search distance and minimum shoreline length. No Hârn-specific coordinates are embedded in the algorithm; the profile still contains image-specific seed points and crop regions. All GeoJSON remains in **image pixel coordinates**, and all features are **unreviewed**. A black stroke may be misidentified as coastline. Lakes, islands, open map borders and narrow channels require manual inspection, especially on the photographed reference.

Synthetic smoke tests (no copyrighted maps):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s MapExtractor -p "test_ink_coast.py"
```

These tests and code have been published but **a full local v0.10 run against the user's photograph has not yet been verified**. Compare the result with v0.7 and v0.9 before adopting this mask for 3D generation. No original Hârn image is stored in the public repo.

## Previous version (v0.9)

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
