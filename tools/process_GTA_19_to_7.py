#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_labels_19_to_7.py
"""

from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

INPUT_DIR  = Path("/GTAV/labels_19")
OUTPUT_DIR = Path("/GTAV/labels_7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IGNORE_LABEL = 255

classes_19 = [
  "road", "sidewalk", "building", "wall", "fence", "pole",
  "traffic light", "traffic sign", "vegetation", "terrain",
  "sky", "person", "rider", "car", "truck", "bus",
  "train", "motorcycle", "bicycle"
]

classes_7 = [
  "road", "sidewalk", "building", "vegetation", "sky", "person", "car"
]

mapping = {i: IGNORE_LABEL for i in range(len(classes_19))}
for i, cname in enumerate(classes_19):
    if cname in classes_7:
        mapping[i] = classes_7.index(cname)

print("Class ID:")
for k, v in mapping.items():
    print(f"{k:2d} ({classes_19[k]:15s}) -> {v}")

for p in tqdm(sorted(INPUT_DIR.glob("*.png"))):
    lbl = np.array(Image.open(p))
    out = np.full_like(lbl, IGNORE_LABEL)

    for old_id, new_id in mapping.items():
        out[lbl == old_id] = new_id

    out_img = Image.fromarray(out.astype(np.uint8), mode="L")
    out_img.save(OUTPUT_DIR / p.name)

print("Out：", OUTPUT_DIR)
