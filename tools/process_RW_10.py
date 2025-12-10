#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# ======================
# Paths & Configurations
# ======================
ROOT = Path("/data1/zd/ROADWork_Data/gtFine")        # Input label root directory (contains train/val/...)
SPLITS = ["val"]                                     # Splits to be processed
OUT_ROOT = Path("/data1/zd/ROADWork_Data/gtFine_10") # Output directory
OUT_ROOT.mkdir(parents=True, exist_ok=True)

IGNORE_LABEL = 255

# ======================
# Final classes (order = train_id)
# Based on merging rules → 10 classes
# ======================
CLASSES = [
    "road",            # 0
    "sidewalk",        # 1  (sidewalk + bike lane + off-road + roadside)
    "barrier",         # 2  (barrier + barricade + fence)
    "police vehicle",  # 3
    "work vehicle",    # 4  (work vehicle + work equipment)
    "police officer",  # 5
    "worker",          # 6
    "cone",            # 7  (cone + drum + vertical panel + tubular marker)
    "arrow board",     # 8
    "ttc sign",        # 9
]

# ======================
# Original IDs (19 classes + 0)
#
# 0: unlabeled
# 1: Road
# 2: Sidewalk
# 3: Bike Lane
# 4: Off-Road
# 5: Roadside
# 6: Barrier
# 7: Barricade
# 8: Fence
# 9: Police Vehicle
# 10: Work Vehicle
# 11: Police Officer
# 12: Worker
# 13: Cone
# 14: Drum
# 15: Vertical Panel
# 16: Tubular Marker
# 17: Work Equipment
# 18: Arrow Board
# 19: TTC Sign
# ======================

# ======================
# Fixed mapping: original id → new train_id
# ======================
ID2TRAIN = {
    0:  IGNORE_LABEL,  # unlabeled -> ignore
    1:  0,  # Road -> road
    2:  1,  # Sidewalk -> sidewalk
    3:  1,  # Bike Lane -> sidewalk
    4:  1,  # Off-Road -> sidewalk
    5:  1,  # Roadside -> sidewalk
    6:  2,  # Barrier -> barrier
    7:  2,  # Barricade -> barrier
    8:  2,  # Fence -> barrier
    9:  3,  # Police Vehicle -> police vehicle
    10: 4,  # Work Vehicle -> work vehicle
    11: 5,  # Police Officer -> police officer
    12: 6,  # Worker -> worker
    13: 7,  # Cone -> cone
    14: 7,  # Drum -> cone
    15: 7,  # Vertical Panel -> cone
    16: 7,  # Tubular Marker -> cone
    17: 4,  # Work Equipment -> work vehicle
    18: 8,  # Arrow Board -> arrow board
    19: 9,  # TTC Sign -> ttc sign
}

ORIG_ID2NAME = {
    0:  "unlabeled",
    1:  "Road",
    2:  "Sidewalk",
    3:  "Bike Lane",
    4:  "Off-Road",
    5:  "Roadside",
    6:  "Barrier",
    7:  "Barricade",
    8:  "Fence",
    9:  "Police Vehicle",
    10: "Work Vehicle",
    11: "Police Officer",
    12: "Worker",
    13: "Cone",
    14: "Drum",
    15: "Vertical Panel",
    16: "Tubular Marker",
    17: "Work Equipment",
    18: "Arrow Board",
    19: "TTC Sign",
}

def dump_classes_json(out_dir: Path):
    """Export classes.json containing name->id and id->name mappings"""
    name2id = {name: i for i, name in enumerate(CLASSES)}
    id2name = {i: name for i, name in enumerate(CLASSES)}
    obj = {"classes": CLASSES, "name2id": name2id, "id2name": id2name}
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def convert_split(split: str):
    in_dir = ROOT / split
    out_dir = OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(in_dir.rglob("*_labelIds.png"))
    print(f"[{split}] Found {len(label_paths)} label files")

    # Build 256-length LUT
    lut = np.full(256, IGNORE_LABEL, dtype=np.uint8)
    for k, v in ID2TRAIN.items():
        lut[int(k)] = int(v)

    for lp in tqdm(label_paths):
        label = np.array(Image.open(lp), dtype=np.uint8)
        label_train = lut[label]

        rel_path = lp.relative_to(in_dir)
        out_path = (out_dir / rel_path).with_name(rel_path.name.replace("_labelIds.png", "_trainIds.png"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(label_train).save(out_path)

    print(f"✅ [{split}] Conversion completed. Saved to: {out_dir}")

def pretty_print_mapping():
    print("\n====== Final Class List (order = train_id) ======")
    for i, n in enumerate(CLASSES):
        print(f"{i:2d}: {n}")

    print("\n====== Original ID -> New train_id Mapping ======")
    for k in sorted(ORIG_ID2NAME.keys()):
        new_id = ID2TRAIN[k]
        tag = "(ignore)" if new_id == IGNORE_LABEL else f"-> {new_id:2d} ({CLASSES[new_id]})"
        print(f"id {k:2d} ({ORIG_ID2NAME[k]:>15s}) {tag}")

if __name__ == "__main__":
    pretty_print_mapping()
    dump_classes_json(OUT_ROOT)
    for s in SPLITS:
        convert_split(s)
    print(f"\nNumber of final classes: {len(CLASSES)} (train_id = 0..{len(CLASSES)-1}, ignore = {IGNORE_LABEL})")
    print(f"classes.json has been saved to: {OUT_ROOT}")
