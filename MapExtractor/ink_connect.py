"""CityVisualizer v0.13 – HDR ink line connectivity.

Find short gaps between facing skeleton endpoints. Keep original ink, accepted
bridges, and coast-proximity selection separate. Never alter v0.7 water.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, mapping

from extract import read_image, save_png
from land_boundary_dual import ink_mask, color_mask
from shoreline import v07_mask
from water import spatial_mask

NEIGHBORS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if dx or dy]


def skeletonize(mask, max_iterations=120):
    """Zhang-Suen thinning using NumPy; black background, white lines."""
    a = np.pad((mask > 0).astype(np.uint8), 1)
    for _ in range(max_iterations):
        removed = 0
        for phase in (0, 1):
            p = a[1:-1, 1:-1]
            n = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:],
                 a[2:, 2:], a[2:, 1:-1], a[2:, :-2],
                 a[1:-1, :-2], a[:-2, :-2]]
            total = sum(n)
            turns = sum((n[j] == 0) & (n[(j + 1) % 8] > 0)
                        for j in range(8))
            if phase == 0:
                condition = (n[0] * n[2] * n[4] == 0) & (
                    n[2] * n[4] * n[6] == 0)
            else:
                condition = (n[0] * n[2] * n[6] == 0) & (
                    n[0] * n[4] * n[6] == 0)
            delete = (p > 0) & (total >= 2) & (total <= 6) & (
                turns == 1) & condition
            removed += int(np.count_nonzero(delete))
            p[delete] = 0
        if removed == 0:
            break
    return a[1:-1, 1:-1] * 255


def neighbor_count(mask):
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    return cv2.filter2D((mask > 0).astype(np.uint8), -1, kernel,
                        borderType=cv2.BORDER_CONSTANT)


def skeleton_components(skeleton):
    return cv2.connectedComponentsWithStats(
        (skeleton > 0).astype(np.uint8), connectivity=8)


def endpoint_direction(labels, component, x, y, radius):
    h, w = labels.shape
    x1, x2 = max(0, x - radius), min(w, x + radius + 1)
    y1, y2 = max(0, y - radius), min(h, y + radius + 1)
    yy, xx = np.where(labels[y1:y2, x1:x2] == component)
    if len(xx) <= 1:
        return None
    vectors = np.column_stack((xx + x1 - x, yy + y1 - y)).astype(float)
    vectors = vectors[np.linalg.norm(vectors, axis=1) > 0]
    if not len(vectors):
        return None
    inward = np.mean(vectors, axis=0)
    magnitude = np.linalg.norm(inward)
    return -inward / magnitude if magnitude > 0.5 else None


def propose_bridges(skeleton, corridor, options):
    maximum = int(options.get("max_gap_px", 12))
    tangent_radius = int(options.get("direction_radius_px", 6))
    threshold = float(options.get("min_direction_cos", 0.45))
    if not 1 <= maximum <= 30 or not 2 <= tangent_radius <= 20:
        raise ValueError("Gap or direction radius is outside supported range")
    components, labels, stats, _ = skeleton_components(skeleton)
    endpoints = np.argwhere((skeleton > 0) & (neighbor_count(skeleton) == 1))
    grid = defaultdict(list)
    records = []
    for y, x in endpoints:
        ident = int(labels[y, x])
        if ident == 0 or not corridor[y, x]:
            continue
        direction = endpoint_direction(labels, ident, int(x), int(y),
                                       tangent_radius)
        if direction is None:
            continue
        record = (int(x), int(y), ident, direction)
        index = len(records)
        records.append(record)
        grid[(int(x)//maximum, int(y)//maximum)].append(index)
    possibilities = []
    for a, (x, y, component, direction) in enumerate(records):
        gx, gy = x//maximum, y//maximum
        for xx in range(gx-1, gx+2):
            for yy in range(gy-1, gy+2):
                for b in grid.get((xx, yy), []):
                    if b <= a:
                        continue
                    bx, by, other_component, other_direction = records[b]
                    if component == other_component:
                        continue
                    delta = np.array([bx-x, by-y], dtype=float)
                    distance = np.linalg.norm(delta)
                    if not 1 < distance <= maximum:
                        continue
                    unit = delta / distance
                    angle_a = float(np.dot(direction, unit))
                    angle_b = float(np.dot(other_direction, -unit))
                    if min(angle_a, angle_b) < threshold:
                        continue
                    sample = np.rint(np.linspace([x, y], [bx, by],
                                                  max(3, int(distance)*2))).astype(int)
                    if not np.all(corridor[sample[:, 1], sample[:, 0]]):
                        continue
                    possibilities.append((distance + 3*(2-angle_a-angle_b),
                                          a, b, angle_a, angle_b))
    possibilities.sort()
    accepted, used = [], set()
    # Union-find prevents bridges from creating loops through unrelated marks.
    parent = list(range(components))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for score, a, b, ca, cb in possibilities:
        if a in used or b in used:
            continue
        x, y, first, _ = records[a]
        bx, by, second, _ = records[b]
        if root(first) == root(second):
            continue
        used.update((a, b))
        parent[root(first)] = root(second)
        accepted.append(((x, y), (bx, by), round(float(score), 2)))
    bridge_mask = np.zeros_like(skeleton)
    for first, last, _ in accepted:
        cv2.line(bridge_mask, first, last, 255, 1)
    return bridge_mask, accepted, len(records)


def coast_corridor(image, water, allowed, profile, options):
    """Broad proximity only; not a definition of coastline."""
    distance = cv2.distanceTransform((water == 0).astype(np.uint8),
                                     cv2.DIST_L2, 3)
    coastal = (distance <= float(options.get("max_water_distance_px", 110)))
    ranges = options.get("shore_hsv_ranges", [])
    if ranges:
        sand = color_mask(image, ranges)
        radius = int(options.get("shore_evidence_radius_px", 18))
        if radius:
            sand = cv2.dilate(sand, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))
        coastal &= sand > 0
    coastal &= allowed > 0
    return coastal


def candidate_components(skeleton, corridor, options):
    count, labels, stats, _ = skeleton_components(skeleton)
    chosen = np.zeros_like(skeleton)
    minimum = int(options.get("min_component_pixels", 25))
    support = float(options.get("min_corridor_fraction", 0.2))
    diagnostic = []
    for i in range(1, count):
        size = int(stats[i, cv2.CC_STAT_AREA])
        if size < minimum:
            continue
        ys, xs = np.where(labels == i)
        fraction = float(np.mean(corridor[ys, xs]))
        diagnostic.append({"component": i, "pixels": size,
                           "coastal_fraction": round(fraction, 3),
                           "selected": fraction >= support})
        if fraction >= support:
            chosen[ys, xs] = 255
    return chosen, diagnostic


def trace_chains(skeleton, min_length=8):
    """Trace skeleton graph edges, split at junctions; keep closed cycles."""
    ys, xs = np.where(skeleton > 0)
    pixels = {(int(x), int(y)) for y, x in zip(ys, xs)}
    neighbors = {point: [q for dx, dy in NEIGHBORS
                         if (q := (point[0]+dx, point[1]+dy)) in pixels]
                 for point in pixels}
    visited = set()
    def edge(a, b):
        return (a, b) if a < b else (b, a)
    def walk(start, successor):
        chain = [start, successor]
        visited.add(edge(start, successor))
        prev, current = start, successor
        while True:
            if len(neighbors[current]) != 2:
                break
            options = [p for p in neighbors[current] if p != prev]
            if not options:
                break
            nxt = options[0]
            e = edge(current, nxt)
            if e in visited:
                break
            visited.add(e)
            chain.append(nxt)
            prev, current = current, nxt
            if current == start:
                break
        return chain
    chains = []
    for start in sorted(neighbors):
        if len(neighbors[start]) == 2:
            continue
        for nxt in neighbors[start]:
            if edge(start, nxt) not in visited:
                chains.append(walk(start, nxt))
    for start in sorted(neighbors):
        for nxt in neighbors[start]:
            if edge(start, nxt) not in visited:
                chains.append(walk(start, nxt))
    lines = []
    for chain in chains:
        if len(chain) < 2:
            continue
        line = LineString(chain)
        if line.length >= min_length:
            lines.append(line)
    return lines


def run(original, hdr, profile, options):
    if original.shape != hdr.shape:
        raise ValueError("HDR and original images must be pixel-aligned and same size")
    allowed = spatial_mask(hdr.shape, profile)
    water = v07_mask(original, profile)
    raw = ink_mask(hdr, profile["land_boundary_v012"])
    raw &= allowed
    corridor = coast_corridor(hdr, water, allowed, profile, options)
    thinned = skeletonize(raw)
    bridges, accepted, endpoints = propose_bridges(thinned, corridor, options)
    connected = cv2.bitwise_or(thinned, bridges)
    selected, diagnostics = candidate_components(connected, corridor, options)
    return {"ink_hdr_raw": raw, "ink_hdr_skeleton": thinned,
            "coastal_corridor": (corridor.astype(np.uint8)*255),
            "bridges": bridges, "ink_hdr_connected": connected,
            "coastal_selection": selected, "water_v07_reference": water}, accepted, endpoints, diagnostics


def main():
    parser = argparse.ArgumentParser(description="v0.13 HDR ink gaps and coast selection")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--hdr", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/ink_connect_v013"))
    args = parser.parse_args()
    original, hdr = read_image(args.image), read_image(args.hdr)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    options = profile.get("ink_connect_v013", {})
    masks, bridges, endpoints, diagnostic = run(original, hdr, profile, options)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        save_png(args.output / (name + ".png"), mask)
    for name, selection in (("all_connected", masks["ink_hdr_connected"]),
                            ("coastal_candidates", masks["coastal_selection"])):
        overlay = hdr.copy()
        overlay[selection > 0] = (0, 185, 245)
        overlay[masks["bridges"] > 0] = (210, 40, 220)
        save_png(args.output / (name + "_overlay.png"), overlay)
    lines = trace_chains(masks["coastal_selection"],
                         float(options.get("min_trace_length_px", 8)))
    features = []
    for i, line in enumerate(lines, 1):
        ident = f"LB{i:04d}"
        features.append({"type": "Feature", "id": ident, "properties": {
            "id": ident, "classification": "land_boundary_candidate",
            "review_status": "unreviewed",
            "method": "hdr_ink_graph_v013",
            "coordinate_system": "image_pixels_top_left_origin"},
            "geometry": mapping(line)})
    (args.output / "land_boundary_candidates_v013.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features},
                   indent=2), encoding="utf-8")
    (args.output / "bridge_proposals_v013.json").write_text(
        json.dumps([{"from": a, "to": b, "score": score}
                    for a, b, score in bridges], indent=2), encoding="utf-8")
    stats = {"version": "0.13", "raw_ink_pixels": int(np.count_nonzero(masks["ink_hdr_raw"])),
             "bridge_count": len(bridges), "endpoint_candidates": endpoints,
             "connected_pixels": int(np.count_nonzero(masks["ink_hdr_connected"])),
             "coastal_selection_pixels": int(np.count_nonzero(masks["coastal_selection"])),
             "candidate_lines": len(lines),
             "warning": "Ink graph candidates, not validated coastlines. All connections unreviewed."}
    (args.output / "stats_v013.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
