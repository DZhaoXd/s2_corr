# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

OPEN30 = [
    "road", "sidewalk", "building", "wall", "bridge", "tunnel",
    "traffic sign", "traffic light", "pole", "fence", "sky",
    "vegetation", "terrain", "water", "snow", "sand",
    "person", "rider", "car", "truck", "bus", "train",
    "bicycle", "motorcycle", "animal", "signboard",
    "railway", "boat", "chair", "trash can"
]
NAME2ID = {name.lower(): i for i, name in enumerate(OPEN30)}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
LABEL_EXTS = (".png", ".tif", ".tiff")
DEFAULT_TARGET_CLASSES = ["railway", "chair", "tunnel", "sand"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Mapillary OV_30 subset from training labels.")
    parser.add_argument("mapillary_path", nargs="?", default="data/mapillary")
    parser.add_argument(
        "--target-classes", nargs="+", default=DEFAULT_TARGET_CLASSES)
    parser.add_argument("--src-label-dir", default=None)
    parser.add_argument("--src-image-dir", default=None)
    parser.add_argument("--dst-label-dir", default=None)
    parser.add_argument("--dst-image-dir", default=None)
    return parser.parse_args()


def find_image(src_image_dir, rel_label):
    base = rel_label.with_suffix("")
    for ext in IMAGE_EXTS:
        path = src_image_dir / base.with_suffix(ext)
        if path.exists():
            return path
    return None


def load_label_array(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[..., 0]
        elif np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(
                arr[..., 1], arr[..., 2]):
            arr = arr[..., 0]
        else:
            raise ValueError(f"Label appears to be RGB, not TrainID: {path}")
    return arr.astype(np.int32, copy=False)


def main():
    args = parse_args()
    root = Path(args.mapillary_path)
    src_label_dir = Path(args.src_label_dir) if args.src_label_dir else (
        root / "training" / "labels_TrainID30")
    src_image_dir = Path(args.src_image_dir) if args.src_image_dir else (
        root / "training" / "images")
    dst_label_dir = Path(args.dst_label_dir) if args.dst_label_dir else (
        root / "OV_30" / "labels")
    dst_image_dir = Path(args.dst_image_dir) if args.dst_image_dir else (
        root / "OV_30" / "images")

    target_ids = [NAME2ID[name.lower()] for name in args.target_classes]
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    dst_image_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(
        p for p in src_label_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in LABEL_EXTS
        and p.name != "open30_classes.txt")
    if not labels:
        raise FileNotFoundError(f"No labels found in {src_label_dir}")

    selected = []
    print(f"[INFO] Checking {len(labels)} Mapillary training labels...")
    for label_path in tqdm(labels):
        rel_label = label_path.relative_to(src_label_dir)
        try:
            arr = load_label_array(label_path)
        except Exception as exc:
            tqdm.write(f"[WARN] Skip {label_path}: {exc}")
            continue

        if not np.any(np.isin(arr, target_ids)):
            continue

        img_path = find_image(src_image_dir, rel_label)
        if img_path is None:
            tqdm.write(f"[WARN] Missing image for label: {rel_label}")
            continue

        dst_lab = dst_label_dir / rel_label
        dst_img = dst_image_dir / img_path.relative_to(src_image_dir)
        dst_lab.parent.mkdir(parents=True, exist_ok=True)
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_path, dst_lab)
        shutil.copy2(img_path, dst_img)
        selected.append(str(rel_label.with_suffix("")))

    print(f"[DONE] Selected {len(selected)} samples.")
    print(f"       Images: {dst_image_dir}")
    print(f"       Labels: {dst_label_dir}")
    print(f"       Target classes: {', '.join(args.target_classes)}")


if __name__ == "__main__":
    main()
