#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IGNORE_LABEL = 255

CLASSES_19 = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle"
]

CLASSES_7 = [
    "road", "sidewalk", "building", "vegetation", "sky", "person", "car"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build GTA5 labels_7 from labels_19.")
    parser.add_argument(
        "gta_path", nargs="?", default="data/GTA5/GTAV",
        help="GTA5 GTAV root containing labels_19 and images.")
    parser.add_argument("--label19-dir", default="labels_19")
    parser.add_argument("--label7-dir", default="labels_7")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.gta_path) / args.label19_dir
    output_dir = Path(args.gta_path) / args.label7_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = {i: IGNORE_LABEL for i in range(len(CLASSES_19))}
    for i, cname in enumerate(CLASSES_19):
        if cname in CLASSES_7:
            mapping[i] = CLASSES_7.index(cname)

    print("Class ID:")
    for k, v in mapping.items():
        print(f"{k:2d} ({CLASSES_19[k]:15s}) -> {v}")

    files = sorted(input_dir.rglob("*.png"))
    if not files:
        raise FileNotFoundError(
            f"No GTA5 19-class labels found in {input_dir}")

    for p in tqdm(files, desc="labels_19 -> labels_7"):
        rel = p.relative_to(input_dir)
        lbl = np.array(Image.open(p))
        out = np.full_like(lbl, IGNORE_LABEL)

        for old_id, new_id in mapping.items():
            out[lbl == old_id] = new_id

        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        out_img = Image.fromarray(out.astype(np.uint8), mode="L")
        out_img.save(dst)

    print("Out:", output_dir)


if __name__ == "__main__":
    main()
