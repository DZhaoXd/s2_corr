# projects/CATSeg/data/register_cityscapes7.py
import os
import json  # <-- 修复1：别忘了导入
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog

SEEN_IDS = {0, 1, 2, 8, 10, 11, 13}
UNSEEN_IDS = [i for i in range(19) if i not in SEEN_IDS]

DEFAULT_SEEN_CLASS_NAMES = [
    "road", "sidewalk", "building", "vegetation", "sky", "person", "car"
]

CITYSCAPES_19 = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle"
]

def _ensure_gt7_from_gt19(gt19_split: str, gt7_split: str):
    gt19_split = Path(gt19_split)
    gt7_split  = Path(gt7_split)
    need_build = (not gt7_split.exists()) or (not any(gt7_split.rglob("*.png")))
    if not need_build:
        return

    for p in gt19_split.rglob("*_gtFine_labelTrainIds.png"):
        rel = p.relative_to(gt19_split)  # e.g. aachen/xxx_gtFine_labelTrainIds.png
        dst = gt7_split / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        m = np.array(Image.open(p), dtype=np.uint8)
        out = np.full_like(m, 255, dtype=np.uint8)
        mask = np.isin(m, list(SEEN_IDS))
        out[mask] = m[mask]
        if not dst.exists():
            Image.fromarray(out).save(dst)

def _load_cityscapes_split(img_root: str, gt_root: str) -> List[Dict]:
    img_root = Path(img_root); gt_root = Path(gt_root)
    dataset: List[Dict] = []
    for img_path in sorted(img_root.rglob("*_leftImg8bit.png")):
        rel_city = img_path.parent.relative_to(img_root)
        stem = img_path.name.replace("_leftImg8bit.png", "")
        gt_path = gt_root / rel_city / f"{stem}_gtFine_labelTrainIds.png"
        if gt_path.exists():
            dataset.append({
                "file_name": str(img_path),
                "sem_seg_file_name": str(gt_path),
                "image_id": f"{rel_city}/{stem}",
            })
    return dataset

def register_cityscapes7_train(
    name: str,
    leftimg_train: str,
    gt19_train: str,
    gt7_train: str,
    class_names: Optional[List[str]] = None,
    class_names_json: Optional[str] = None,
):
    _ensure_gt7_from_gt19(gt19_train, gt7_train)

    if class_names is None:
        if class_names_json is not None and os.path.isfile(class_names_json):
            with open(class_names_json, "r") as f:
                class_names = json.load(f)
        else:
            class_names = DEFAULT_SEEN_CLASS_NAMES

    DatasetCatalog.register(
        name,
        lambda a=leftimg_train, b=gt7_train: _load_cityscapes_split(a, b)
    )
    MetadataCatalog.get(name).set(
        stuff_classes=class_names,
        evaluator_type="sem_seg",
        ignore_label=255,
        image_root=leftimg_train,
        sem_seg_root=gt7_train,
        seen_ids=sorted(list(SEEN_IDS)),
        unseen_ids=UNSEEN_IDS,
    )

def register_all_cityscapes7(root="/data/zd/data/cityscape/cityscapes/Cityscapes/"):
    left_train = os.path.join(root, "leftImg8bit/train")
    gt19_train = os.path.join(root, "gtFine_19/train")
    gt7_train  = os.path.join(root, "gtFine_7/train")
    register_cityscapes7_train(
        "cityscapes7_train", left_train, gt19_train, gt7_train,
        class_names=None,
        class_names_json="datasets/cityscapes_seen_7.json",
    )

register_all_cityscapes7("/data/zd/data/cityscape/cityscapes/Cityscapes/")
