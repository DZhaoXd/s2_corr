# projects/CATSeg/data/register_cityscapes19.py
import os
from pathlib import Path
from typing import List, Dict

from detectron2.data import DatasetCatalog, MetadataCatalog

CITYSCAPES_19 = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle"
]

def _load_cityscapes19_split(leftimg8bit_split_root: str,
                             gtfine19_split_root: str) -> List[Dict]:
    img_root = Path(leftimg8bit_split_root)
    gt_root = Path(gtfine19_split_root)

    dataset: List[Dict] = []
    for img_path in sorted(img_root.rglob("*_leftImg8bit.png")):
        rel_city = img_path.parent.relative_to(img_root)  # e.g. aachen
        stem = img_path.name.replace("_leftImg8bit.png", "")
        gt_name = f"{stem}_gtFine_labelTrainIds.png"
        gt_path = gt_root / rel_city / gt_name
        if not gt_path.exists():
            continue

        dataset.append({
            "file_name": str(img_path),
            "sem_seg_file_name": str(gt_path),
            "image_id": f"{rel_city}/{stem}",
        })
    return dataset


def register_cityscapes19_split(name: str,
                                leftimg8bit_split_root: str,
                                gtfine19_split_root: str,
                                classes=CITYSCAPES_19):
    DatasetCatalog.register(
        name,
        lambda a=leftimg8bit_split_root, b=gtfine19_split_root: _load_cityscapes19_split(a, b)
    )
    MetadataCatalog.get(name).set(
        stuff_classes=classes,
        evaluator_type="sem_seg",
        ignore_label=255,
        image_root=leftimg8bit_split_root,
        sem_seg_root=gtfine19_split_root,
    )


def register_all_cityscapes19(root="/data/zd/data/cityscape/cityscapes/Cityscapes/"):
    # train
    left_train = os.path.join(root, "leftImg8bit/train")
    gt19_train = os.path.join(root, "gtFine_19/train")
    register_cityscapes19_split("cityscapes19_train", left_train, gt19_train)

    # val（若有 19 类标签）
    left_val = os.path.join(root, "leftImg8bit/val")
    gt19_val = os.path.join(root, "gtFine_19/val")
    if os.path.isdir(left_val) and os.path.isdir(gt19_val):
        register_cityscapes19_split("cityscapes19_val", left_val, gt19_val)

register_all_cityscapes19("/data/zd/data/cityscape/cityscapes/Cityscapes/")
