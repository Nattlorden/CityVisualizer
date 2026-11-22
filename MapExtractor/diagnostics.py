"""MapExtractor v0.3: diagnostic crops from unchanged v0.2 extraction."""
import argparse
import json
from pathlib import Path
import cv2
from extract import read_image, save_png, extract, failed_filters

DEFAULT_REGIONS = {
    "east_small_buildings": [1090, 315, 1495, 550],
    "central_city": [295, 270, 620, 830],
}

def valid_rect(rect, width, height):
    x1, y1, x2, y2 = map(int, rect)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop: {rect}")
    return x1, y1, x2, y2

def within(item, rect):
    x, y, w, h = item["bbox"]
    cx, cy = x + w / 2, y + h / 2
    return rect[0] <= cx < rect[2] and rect[1] <= cy < rect[3]

def overlay(image, baseline, additions, rejected, rect, mode):
    canvas = image.copy()
    for item in baseline:
        if within(item, rect):
            cv2.drawContours(canvas, [item["contour"]], -1, (40, 40, 225), 2)
    if mode == "v03":
        for group, color in ((additions, (0, 215, 255)), (rejected, (255, 80, 200))):
            for item in group:
                if within(item, rect):
                    cv2.drawContours(canvas, [item["contour"]], -1, color, 1)
                    x, y, w, h = item["bbox"]
                    cv2.rectangle(canvas, (max(0, x-2), max(0, y-2)),
                                  (x+w+2, y+h+2), color, 1)
    x1, y1, x2, y2 = rect
    return cv2.resize(canvas[y1:y2, x1:x2], ((x2-x1)*3, (y2-y1)*3),
                      interpolation=cv2.INTER_NEAREST)

def main():
    p = argparse.ArgumentParser(description="v0.3 diagnostics: baseline, additions and rejections")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("MapExtractor/sample-config.json"))
    p.add_argument("--output", type=Path, default=Path("output/diagnostics_v03"))
    args = p.parse_args()
    image = read_image(args.image)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    mask, baseline, additions, _, contour_count = extract(image, config)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted = {item["contour"].tobytes() for item in baseline + additions}
    rejects = []
    for contour in contours:
        if contour.tobytes() in accepted:
            continue
        area = float(cv2.contourArea(contour))
        bbox = cv2.boundingRect(contour)
        reasons = failed_filters(area, bbox, config["v02_filters"])
        rejects.append({"contour": contour, "bbox": bbox, "area": area,
                        "reasons": reasons or ["invalid_polygon"]})
    args.output.mkdir(parents=True, exist_ok=True)
    regions = config.get("diagnostic_regions", DEFAULT_REGIONS)
    report = {"baseline": len(baseline), "added_v02": len(additions),
              "still_rejected": len(rejects), "mask_contours": contour_count, "regions": {}}
    for name, coords in regions.items():
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Unsafe region name: {name}")
        rect = valid_rect(coords, image.shape[1], image.shape[0])
        counts = {"baseline": sum(within(i, rect) for i in baseline),
                  "added_v02": sum(within(i, rect) for i in additions),
                  "rejected": sum(within(i, rect) for i in rejects)}
        report["regions"][name] = {"crop": list(rect), **counts}
        for mode in ("baseline", "v03"):
            save_png(args.output / f"{name}_{mode}.png",
                     overlay(image, baseline, additions, rejects, rect, mode))
        print(f"{name}: {counts}")
    (args.output / "diagnostic_stats.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved diagnostics: {args.output.resolve()}")

if __name__ == "__main__":
    main()
