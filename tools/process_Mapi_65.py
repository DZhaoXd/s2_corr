#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IGNORE_LABEL = 255
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

OPEN30 = [
    "road", "sidewalk", "building", "wall", "bridge", "tunnel",
    "traffic sign", "traffic light", "pole", "fence", "sky",
    "vegetation", "terrain", "water", "snow", "sand",
    "person", "rider", "car", "truck", "bus", "train",
    "bicycle", "motorcycle", "animal", "signboard",
    "railway", "boat", "chair", "trash can"
]
OPEN30_ID = {n: i for i, n in enumerate(OPEN30)}

CS_ALIGNED = {
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle"
}
CS19_PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32)
]
CS_NAME2IDX = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
    "pole": 5, "traffic light": 6, "traffic sign": 7,
    "vegetation": 8, "terrain": 9, "sky": 10, "person": 11,
    "rider": 12, "car": 13, "truck": 14, "bus": 15, "train": 16,
    "motorcycle": 17, "bicycle": 18
}

EXTRA_COLORS = [
    (255, 165, 0), (64, 224, 208), (123, 104, 238), (46, 139, 87),
    (199, 21, 133), (95, 158, 160), (205, 133, 63),
    (176, 196, 222), (30, 144, 255), (65, 105, 225), (210, 105, 30),
]

M0_65_TO_O30 = np.full(256, IGNORE_LABEL, dtype=np.uint8)
PAIRS = {
    0: OPEN30_ID["animal"], 1: OPEN30_ID["animal"],
    2: OPEN30_ID["sidewalk"], 3: OPEN30_ID["fence"],
    4: OPEN30_ID["fence"], 5: OPEN30_ID["fence"],
    6: OPEN30_ID["wall"], 7: OPEN30_ID["road"],
    8: OPEN30_ID["sidewalk"], 9: OPEN30_ID["sidewalk"],
    10: OPEN30_ID["terrain"], 11: OPEN30_ID["sidewalk"],
    12: OPEN30_ID["railway"], 13: OPEN30_ID["road"],
    14: OPEN30_ID["road"], 15: OPEN30_ID["sidewalk"],
    16: OPEN30_ID["bridge"], 17: OPEN30_ID["building"],
    18: OPEN30_ID["tunnel"], 19: OPEN30_ID["person"],
    20: OPEN30_ID["rider"], 21: OPEN30_ID["rider"],
    22: OPEN30_ID["rider"], 23: OPEN30_ID["sidewalk"],
    24: OPEN30_ID["road"], 25: OPEN30_ID["terrain"],
    26: OPEN30_ID["sand"], 27: OPEN30_ID["sky"],
    28: OPEN30_ID["snow"], 29: OPEN30_ID["terrain"],
    30: OPEN30_ID["vegetation"], 31: OPEN30_ID["water"],
    32: OPEN30_ID["signboard"], 33: OPEN30_ID["chair"],
    34: OPEN30_ID["fence"], 35: OPEN30_ID["signboard"],
    36: IGNORE_LABEL, 37: IGNORE_LABEL, 38: IGNORE_LABEL,
    39: IGNORE_LABEL, 40: IGNORE_LABEL, 41: IGNORE_LABEL,
    42: IGNORE_LABEL, 43: OPEN30_ID["road"], 44: OPEN30_ID["pole"],
    45: OPEN30_ID["pole"], 46: OPEN30_ID["signboard"],
    47: OPEN30_ID["pole"], 48: OPEN30_ID["traffic light"],
    49: OPEN30_ID["traffic sign"], 50: OPEN30_ID["traffic sign"],
    51: OPEN30_ID["trash can"], 52: OPEN30_ID["bicycle"],
    53: OPEN30_ID["boat"], 54: OPEN30_ID["bus"],
    55: OPEN30_ID["car"], 56: OPEN30_ID["truck"],
    57: OPEN30_ID["motorcycle"], 58: OPEN30_ID["train"],
    59: IGNORE_LABEL, 60: OPEN30_ID["truck"], 61: OPEN30_ID["truck"],
    62: IGNORE_LABEL, 63: IGNORE_LABEL, 64: IGNORE_LABEL,
    65: IGNORE_LABEL,
}
for old_id, new_id in PAIRS.items():
    M0_65_TO_O30[old_id] = new_id


def build_palette():
    pal = [128, 128, 128] * 256
    extra_i = 0
    for i, name in enumerate(OPEN30):
        if name in CS_ALIGNED:
            color = CS19_PALETTE[CS_NAME2IDX[name]]
        else:
            color = EXTRA_COLORS[extra_i % len(EXTRA_COLORS)]
            extra_i += 1
        pal[i * 3:i * 3 + 3] = list(color)
    pal[IGNORE_LABEL * 3:IGNORE_LABEL * 3 + 3] = [0, 0, 0]
    return pal


PALETTE = build_palette()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Mapillary Vistas 65 labels to Open-Mapillary-30.")
    parser.add_argument("mapillary_path", nargs="?", default="data/mapillary")
    parser.add_argument(
        "--splits", nargs="+", default=["training", "validation"],
        choices=["training", "validation"])
    parser.add_argument(
        "--image-mode", default="symlink", choices=["symlink", "copy", "none"],
        help="How to create val/images from validation/images.")
    return parser.parse_args()


def remap_label(arr):
    if arr.ndim != 2:
        raise ValueError("Input must be a single-channel indexed label.")
    if arr.dtype != np.uint8:
        if arr.max() < 256:
            arr = arr.astype(np.uint8)
        else:
            raise ValueError("Found label pixel values >255.")
    return M0_65_TO_O30[arr]


def save_indexed_png(path, arr):
    im = Image.fromarray(arr.astype(np.uint8), mode="P")
    im.putpalette(PALETTE)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG", optimize=True)


def convert_split(input_dir, output_dir):
    files = sorted(input_dir.rglob("*.png"))
    if not files:
        raise FileNotFoundError(f"No Mapillary label PNG files found in {input_dir}")

    for p in tqdm(files, desc=f"{input_dir.parent.name} -> labels_TrainID30"):
        with Image.open(p) as im:
            if im.mode not in ("L", "P", "I;16"):
                raise ValueError(
                    f"Detected RGB label: {p}. Use indexed labels instead.")
            arr = np.array(im)
            if arr.dtype != np.uint8 and arr.max() < 256:
                arr = arr.astype(np.uint8)
        save_indexed_png(output_dir / p.relative_to(input_dir),
                         remap_label(arr))

    with open(output_dir / "open30_classes.txt", "w", encoding="utf-8") as f:
        for i, name in enumerate(OPEN30):
            f.write(f"{i}\t{name}\n")


def mirror_images(src_dir, dst_dir, mode):
    if mode == "none":
        return

    files = [
        p for p in src_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not files:
        raise FileNotFoundError(f"No Mapillary images found in {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in tqdm(files, desc=f"images -> {dst_dir}"):
        dst = dst_dir / src.relative_to(src_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue
        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.relpath(src, dst.parent), dst)


def main():
    args = parse_args()
    root = Path(args.mapillary_path)
    for split in args.splits:
        input_dir = root / split / "labels"
        if split == "validation":
            output_dir = root / "val" / "labels_TrainID30"
        else:
            output_dir = root / split / "labels_TrainID30"
        convert_split(input_dir, output_dir)

    if "validation" in args.splits:
        mirror_images(root / "validation" / "images", root / "val" / "images",
                      args.image_mode)

    print("[DONE] Mapillary Open-Mapillary-30 labels are ready.")


if __name__ == "__main__":
    main()
