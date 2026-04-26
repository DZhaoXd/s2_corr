import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

REMAPPING = {
    0: 0,   # road
    1: 1,   # sidewalk
    2: 2,   # building
    8: 3,   # vegetation
    10: 4,  # sky
    11: 5,  # person
    13: 6,  # car
}
IGNORE_VALUE = 255


def process_split(gt19_split, gt7_split):
    gt19_split = Path(gt19_split)
    gt7_split = Path(gt7_split)
    files = sorted(gt19_split.rglob("*_gtFine_labelTrainIds.png"))
    if not files:
        raise FileNotFoundError(
            f"No Cityscapes 19-class labels found in {gt19_split}")

    for p in tqdm(files, desc=f"{gt19_split.name} -> gtFine_7"):
        rel = p.relative_to(gt19_split)
        dst = gt7_split / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        m = np.array(Image.open(p), dtype=np.uint8)
        out = np.full_like(m, IGNORE_VALUE, dtype=np.uint8)
        for old_id, new_id in REMAPPING.items():
            out[m == old_id] = new_id
        Image.fromarray(out).save(dst)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Cityscapes gtFine_7 labels from gtFine_19 labels.")
    parser.add_argument(
        "cityscapes_path", nargs="?", default="data/cityscape",
        help="Cityscapes root containing gtFine_19 and leftImg8bit.")
    parser.add_argument(
        "--splits", nargs="+", default=["train"],
        help="Splits to convert, e.g. train val.")
    parser.add_argument("--gt19-dir", default="gtFine_19")
    parser.add_argument("--gt7-dir", default="gtFine_7")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.cityscapes_path)
    for split in args.splits:
        process_split(root / args.gt19_dir / split,
                      root / args.gt7_dir / split)


if __name__ == "__main__":
    main()
