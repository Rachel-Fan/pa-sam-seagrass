# evaluate_eelgrass_metrics.py
# ==========================================================
# Evaluate segmentation outputs (Pa-SAM / U-Net) for eelgrass dataset
# Author: RF | 2025-10
# ==========================================================

import os
import csv
import json
import time
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from skimage import io, morphology, measure

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
GT_DIR      = r"./dataset/mask"             # ground truth mask folder
PRED_DIR    = r"./outputs/unet_pred"        # predicted mask folder
METADATA    = r"./dataset/metadata.csv"     # optional (site/year/modality)
OUT_CSV     = r"./metrics_summary.csv"      # output summary

PIXEL_SIZE  = 0.05   # (m) example: 5 cm per pixel; update to match metadata
BOUND_TOL   = 3      # pixel tolerance for boundary F1

# ----------------------------------------------------------
# METRICS FUNCTIONS
# ----------------------------------------------------------
def compute_confusion(gt, pred, num_classes=2):
    mask = (gt >= 0) & (gt < num_classes)
    label = num_classes * gt[mask].astype('int') + pred[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion = count.reshape(num_classes, num_classes)
    return confusion

def compute_iou(confusion):
    intersection = np.diag(confusion)
    union = np.sum(confusion, axis=1) + np.sum(confusion, axis=0) - intersection
    iou = intersection / np.maximum(union, 1e-6)
    return np.nanmean(iou), iou

def dice_coef(gt, pred):
    intersection = np.sum(gt * pred)
    return (2. * intersection) / (np.sum(gt) + np.sum(pred) + 1e-6)

def boundary_f1(gt, pred, tol=3):
    gt_bound = morphology.binary_dilation(gt, morphology.disk(tol)) ^ gt
    pr_bound = morphology.binary_dilation(pred, morphology.disk(tol)) ^ pred
    gt_dil = morphology.binary_dilation(gt_bound, morphology.disk(tol))
    pr_dil = morphology.binary_dilation(pr_bound, morphology.disk(tol))
    precision = np.sum(pr_bound & gt_dil) / (np.sum(pr_bound) + 1e-6)
    recall = np.sum(gt_bound & pr_dil) / (np.sum(gt_bound) + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6)

def area_error(gt, pred, pixel_size=0.05):
    """Relative area difference (%)"""
    area_gt = np.sum(gt) * pixel_size**2
    area_pr = np.sum(pred) * pixel_size**2
    return abs(area_gt - area_pr) / (area_gt + 1e-6) * 100

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    start = time.time()
    print(f"[INFO] Evaluating predictions in {PRED_DIR}")
    all_preds = sorted(glob(os.path.join(PRED_DIR, "*.png")))
    results = []

    # optional metadata mapping
    meta = None
    if os.path.exists(METADATA):
        meta = pd.read_csv(METADATA)

    for pred_path in tqdm(all_preds):
        fname = os.path.basename(pred_path)
        gt_path = os.path.join(GT_DIR, fname)
        if not os.path.exists(gt_path):
            continue

        gt = io.imread(gt_path)
        pr = io.imread(pred_path)

        # handle multi-channel or 255-range
        if gt.ndim > 2:
            gt = gt[...,0]
        if pr.ndim > 2:
            pr = pr[...,0]
        gt = (gt > 127).astype(np.uint8)
        pr = (pr > 127).astype(np.uint8)

        # metrics
        conf = compute_confusion(gt, pr)
        miou, class_iou = compute_iou(conf)
        dice = dice_coef(gt, pr)
        bf1 = boundary_f1(gt, pr, tol=BOUND_TOL)
        aerr = area_error(gt, pr, pixel_size=PIXEL_SIZE)

        row = {
            "filename": fname,
            "mIoU": miou,
            "Dice": dice,
            "BoundaryF1": bf1,
            "AreaError_%": aerr,
            "IoU_class0": class_iou[0],
            "IoU_class1": class_iou[1],
        }

        # attach site/year/modality if metadata exists
        if meta is not None and "filename" in meta.columns:
            mrow = meta.loc[meta["filename"] == fname]
            if not mrow.empty:
                for col in ["site", "year", "modality"]:
                    if col in mrow.columns:
                        row[col] = mrow.iloc[0][col]

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Saved per-image metrics → {OUT_CSV}")

    # summary stats
    summary = df.groupby(["site", "year", "modality"], dropna=False)[
        ["mIoU", "Dice", "BoundaryF1", "AreaError_%"]
    ].mean().reset_index()
    summary.to_csv(OUT_CSV.replace(".csv", "_summary.csv"), index=False)
    print(f"[OK] Saved grouped summary → {OUT_CSV.replace('.csv', '_summary.csv')}")

    print(f"[DONE] {len(df)} images evaluated in {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
