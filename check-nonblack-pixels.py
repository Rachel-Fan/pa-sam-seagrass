#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_nonblack_pixels.py
------------------------
Scan a folder of PNG images and report how many pixels are non-black (any channel > 0).

Usage:
  python check_nonblack_pixels.py
  # or specify folder path
  python check_nonblack_pixels.py "/path/to/folder"
"""

import os
import sys
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

def count_nonblack_pixels(img_dir):
    records = []

    for fname in tqdm(sorted(os.listdir(img_dir)), desc="Analyzing"):
        if not fname.lower().endswith(".png"):
            continue
        fpath = os.path.join(img_dir, fname)
        img = np.array(Image.open(fpath))

        # Convert to grayscale if needed
        if img.ndim == 3:
            mask = img.sum(axis=2) > 0
        else:
            mask = img > 0

        total = mask.size
        non_black = mask.sum()
        ratio = non_black / total

        records.append({
            "filename": fname,
            "non_black_pixels": int(non_black),
            "total_pixels": int(total),
            "ratio_non_black": round(ratio, 6)
        })

    df = pd.DataFrame(records)
    avg_ratio = df["ratio_non_black"].mean()
    print(f"\nTotal images: {len(df)}")
    print(f"Average non-black ratio: {avg_ratio:.6f}")
    return df

def main():
    if len(sys.argv) > 1:
        img_dir = sys.argv[1]
    else:
        img_dir = input("Enter image folder path: ").strip() or "./output/2025/All/rgb/train_run2/vis_test-1029-2/predict-only"

    if not os.path.isdir(img_dir):
        print(f"[Error] Folder not found: {img_dir}")
        sys.exit(1)

    df = count_nonblack_pixels(img_dir)
    out_csv = os.path.join(img_dir, "non_black_pixel_stats.csv")
    df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

if __name__ == "__main__":
    main()
