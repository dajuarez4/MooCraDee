#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def circularity(area, perimeter):
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def circle_iou(c1, c2):
    # c = (x,y,r,score)
    x1, y1, r1, _ = c1
    x2, y2, r2, _ = c2
    d = float(np.hypot(x1 - x2, y1 - y2))

    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        inter = np.pi * min(r1, r2) ** 2
        union = np.pi * max(r1, r2) ** 2
        return float(inter / union)

    a1 = r1 * r1 * np.arccos((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1))
    a2 = r2 * r2 * np.arccos((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2))
    a3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    inter = a1 + a2 - a3
    union = np.pi * r1 * r1 + np.pi * r2 * r2 - inter
    return float(inter / union)


def dedup_by_iou(circles, iou_thr=0.15):
    keep = []
    for c in circles:
        ok = True
        for k in keep:
            if circle_iou(c, k) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(c)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Imagen ORIGINAL (sin círculos dibujados).")
    ap.add_argument("--ckpt", required=True, help="Checkpoint SAM .pth (ej: sam_vit_b_01ec64.pth)")
    ap.add_argument("--model", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--out", default="sam_craters_out.png")
    ap.add_argument("--csv", default="sam_craters_radii.csv")

    ap.add_argument("--min_radius", type=float, default=40)
    ap.add_argument("--max_radius", type=float, default=250)
    ap.add_argument("--min_circularity", type=float, default=0.55)  # 0..1 (más alto = más circular)
    ap.add_argument("--min_area", type=int, default=1500)           # máscaras muy pequeñas fuera
    ap.add_argument("--iou_dedup", type=float, default=0.15)

    ap.add_argument("--pps", type=int, default=32, help="points_per_side (más = más máscaras, más lento)")
    ap.add_argument("--pred_iou", type=float, default=0.88)
    ap.add_argument("--stability", type=float, default=0.92)
    ap.add_argument("--min_mask_region_area", type=int, default=1000)

    args = ap.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model](checkpoint=args.ckpt)
    sam.to(device=device)

    mask_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.pps,
        pred_iou_thresh=args.pred_iou,
        stability_score_thresh=args.stability,
        min_mask_region_area=args.min_mask_region_area,
    )

    masks = mask_gen.generate(img_rgb)

    circles = []
    for m in masks:
        area = int(m.get("area", 0))
        if area < args.min_area:
            continue

        seg = m["segmentation"].astype(np.uint8) * 255

        seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        c = max(cnts, key=cv2.contourArea)
        per = float(cv2.arcLength(c, True))
        a = float(cv2.contourArea(c))
        if a <= 0:
            continue

        circ = circularity(a, per)
        if circ < args.min_circularity:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)

        if r < args.min_radius or r > args.max_radius:
            continue
        if x - r < 5 or y - r < 5 or x + r > w - 5 or y + r > h - 5:
            continue

        
        score = float(m.get("stability_score", 0.0)) + 0.5 * float(m.get("predicted_iou", 0.0)) + 0.5 * circ
        circles.append((float(x), float(y), float(r), score))

    # ---------------- sort 
    circles.sort(key=lambda t: t[3], reverse=True)
    circles = dedup_by_iou(circles, iou_thr=args.iou_dedup)

    # ---------------- drawn back
    out = img_bgr.copy()
    for x, y, r, score in circles:
        cv2.circle(out, (int(round(x)), int(round(y))), int(round(r)), (0, 0, 255), 3)
        cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)

    cv2.imwrite(args.out, out)

    # ---------------- csv 
    with open(args.csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["id", "x_px", "y_px", "radius_px", "score"])
        for i, (x, y, r, score) in enumerate(circles, 1):
            wcsv.writerow([i, f"{x:.2f}", f"{y:.2f}", f"{r:.2f}", f"{score:.4f}"])

    print("Device:", device)
    print("Circulos finales:", len(circles))
    for i, (x, y, r, score) in enumerate(circles, 1):
        print(f"{i}: centro=({x:.1f},{y:.1f})  radio={r:.1f}px  score={score:.3f}")


if __name__ == "__main__":
    main()
