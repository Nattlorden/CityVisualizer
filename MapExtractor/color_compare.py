"""CityVisualizer v0.4: compare original RGB extraction with local LAB correction.

This is diagnostic only. It does not replace v0.2 masks, objects or IDs.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from extract import read_image, save_png, make_mask, failed_filters

def local_lab_normalize(image, params):
    sigma = float(params.get("sigma_px", 95))
    strength = float(params.get("strength", 0.65))
    limit_l = float(params.get("max_light_shift", 24))
    limit_ab = float(params.get("max_chroma_shift", 11))
    if sigma <= 1 or not 0 <= strength <= 1 or limit_l < 0 or limit_ab < 0:
        raise ValueError("Invalid normalization settings")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = cv2.GaussianBlur(lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    reference = np.median(background.reshape(-1, 3), axis=0)
    shift = (reference - background) * strength
    shift[:, :, 0] = np.clip(shift[:, :, 0], -limit_l, limit_l)
    shift[:, :, 1:] = np.clip(shift[:, :, 1:], -limit_ab, limit_ab)
    corrected = np.clip(lab + shift, 0, 255).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

def candidates(mask, rules):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted = [c for c in contours if not failed_filters(
        cv2.contourArea(c), cv2.boundingRect(c), rules)]
    return accepted, len(contours)

def crop_images(output, image, rgb, corrected, diff, regions):
    height, width = image.shape[:2]
    for name, coords in regions.items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Unsafe region name")
        x1, y1, x2, y2 = map(int, coords)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid region: {name}")
        for label, data in (("original", image), ("rgb_mask", rgb),
                            ("corrected_mask", corrected), ("difference", diff)):
            cropped = data[y1:y2, x1:x2]
            zoomed = cv2.resize(cropped, None, fx=3, fy=3,
                                interpolation=cv2.INTER_NEAREST)
            save_png(output / f"{name}_{label}.png", zoomed)

def main():
    parser = argparse.ArgumentParser(description="Experimental v0.4 LAB mask comparison")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("MapExtractor/sample-config.json"))
    parser.add_argument("--output", type=Path, default=Path("output/color_compare_v04"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    params = config.get("v04_color_normalization", {})
    image = read_image(args.image)
    original_mask = make_mask(image, config)
    corrected_image = local_lab_normalize(image, params)
    normalized_mask = make_mask(corrected_image, config)
    shared = (original_mask > 0) & (normalized_mask > 0)
    added = (original_mask == 0) & (normalized_mask > 0)
    lost = (original_mask > 0) & (normalized_mask == 0)
    diff = np.full_like(image, (40, 40, 40))
    diff[shared] = (205, 205, 205)
    diff[added] = (0, 225, 30)  # green: new
    diff[lost] = (205, 20, 205)  # magenta: removed
    old_contours, old_total = candidates(original_mask, config["v02_filters"])
    new_contours, new_total = candidates(normalized_mask, config["v02_filters"])
    old_overlay = image.copy()
    new_overlay = image.copy()
    cv2.drawContours(old_overlay, old_contours, -1, (20, 35, 220), 2)
    cv2.drawContours(new_overlay, new_contours, -1, (0, 210, 255), 2)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("original_mask", original_mask),
                       ("normalized_image", corrected_image),
                       ("normalized_mask", normalized_mask),
                       ("mask_difference", diff),
                       ("rgb_candidate_overlay", old_overlay),
                       ("normalized_candidate_overlay", new_overlay)):
        save_png(args.output / (name + ".png"), data)
    regions = config.get("v04_diagnostic_regions", {
        "east_small_buildings": [1090, 315, 1495, 550],
        "central_city": [295, 270, 620, 830]
    })
    crop_images(args.output, image, original_mask, normalized_mask, diff, regions)
    stats = {
        "version": "0.4", "experimental": True,
        "image_pixels": [image.shape[1], image.shape[0]],
        "rgb_mask_white_pixels": int(np.count_nonzero(original_mask)),
        "normalized_mask_white_pixels": int(np.count_nonzero(normalized_mask)),
        "added_white_pixels": int(np.count_nonzero(added)),
        "lost_white_pixels": int(np.count_nonzero(lost)),
        "rgb_contours": old_total, "normalized_contours": new_total,
        "rgb_candidate_contours": len(old_contours),
        "normalized_candidate_contours": len(new_contours),
        "regions": regions,
        "note": "Contour counts are not counts of new buildings: contours can split/merge."
    }
    (args.output / "comparison_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved to: {args.output.resolve()}")

if __name__ == "__main__":
    main()
