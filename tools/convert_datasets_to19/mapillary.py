# ---------------------------------------------------------------
# Copyright (c) 2023 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import json
import os
import shutil
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ID_TO_TRAIN_ID = {
    13: 0,  # Road
    24: 0,  # Lane Marking - General
    41: 0,  # Manhole
    2: 1,  # Curb
    15: 1,  # Sidewalk
    17: 2,  # Building
    6: 3,  # Wall
    3: 4,  # Fence
    45: 5,  # Pole
    47: 5,  # Utility Pole
    48: 6,  # Traffic Light
    50: 7,  # Traffic Sign
    30: 8,  # Vegetation
    29: 9,  # Terrain
    27: 10,  # Sky
    19: 11,  # Person
    20: 12,  # Bicyclist
    21: 12,  # Motorcyclist
    22: 12,  # Other Rider
    55: 13,  # Car
    61: 14,  # Truck
    54: 15,  # Bus
    58: 16,  # On Rails
    57: 17,  # Motorcycle
    52: 18,  # Bicycle
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def convert_to_train_id(file, gt_dir, out_label_dir):
    label = np.asarray(Image.open(file))
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in ID_TO_TRAIN_ID.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n

    rel_file = Path(file).relative_to(gt_dir)
    new_file = Path(out_label_dir) / rel_file
    new_file.parent.mkdir(parents=True, exist_ok=True)
    sample_class_stats["file"] = str(new_file)
    Image.fromarray(label_copy, mode="L").save(str(new_file))
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Mapillary Vistas labels to Cityscapes trainIds.")
    parser.add_argument("mapillary_path", help="Mapillary Vistas data root.")
    parser.add_argument(
        "--split", default="validation", choices=["training", "validation"],
        help="Official Mapillary split to convert.")
    parser.add_argument(
        "--gt-dir", default=None,
        help="Input label directory. Defaults to <split>/labels.")
    parser.add_argument(
        "--out-label-dir", default=None,
        help=("Output label directory. Defaults to val/labels_TrainIds for "
              "validation and <split>/labels_TrainIds otherwise."))
    parser.add_argument(
        "--image-out-dir", default=None,
        help=("Where to mirror images. Defaults to val/images for validation "
              "and does nothing for other splits."))
    parser.add_argument(
        "--image-mode", default="symlink", choices=["symlink", "copy", "none"],
        help="How to create image files in --image-out-dir.")
    parser.add_argument(
        "-o", "--out-dir", default=None,
        help="Directory for sample class-stat JSON files.")
    parser.add_argument(
        "--nproc", default=4, type=int, help="number of processes")
    return parser.parse_args()


def save_class_stats(out_dir, sample_class_stats):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_class_stats = [dict(e) for e in sample_class_stats]
    with open(out_dir / "sample_class_stats.json", "w") as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop("file")
        sample_class_stats_dict[f] = stats
    with open(out_dir / "sample_class_stats_dict.json", "w") as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            samples_with_class.setdefault(c, []).append((file, n))
    with open(out_dir / "samples_with_class.json", "w") as of:
        json.dump(samples_with_class, of, indent=2)


def track_progress(func, files, nproc):
    if nproc > 1:
        with Pool(nproc) as pool:
            return list(tqdm(pool.imap(func, files), total=len(files)))
    return [func(file) for file in tqdm(files)]


def mirror_images(src_dir, dst_dir, mode):
    if mode == "none" or dst_dir is None:
        return

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
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
    mapillary_path = Path(args.mapillary_path)
    gt_dir = Path(args.gt_dir) if args.gt_dir else (
        mapillary_path / args.split / "labels")
    out_label_dir = Path(args.out_label_dir) if args.out_label_dir else (
        mapillary_path / "val" / "labels_TrainIds"
        if args.split == "validation"
        else mapillary_path / args.split / "labels_TrainIds")
    out_dir = Path(args.out_dir) if args.out_dir else out_label_dir.parent

    poly_files = sorted(str(p) for p in gt_dir.rglob("*.png"))
    if not poly_files:
        raise FileNotFoundError(f"No Mapillary label PNG files found in {gt_dir}")

    out_label_dir.mkdir(parents=True, exist_ok=True)
    worker = partial(
        convert_to_train_id, gt_dir=gt_dir, out_label_dir=out_label_dir)
    sample_class_stats = track_progress(worker, poly_files, args.nproc)
    save_class_stats(out_dir, sample_class_stats)

    image_out_dir = args.image_out_dir
    if image_out_dir is None and args.split == "validation":
        image_out_dir = mapillary_path / "val" / "images"
    mirror_images(mapillary_path / args.split / "images", image_out_dir,
                  args.image_mode)


if __name__ == "__main__":
    main()
