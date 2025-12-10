# register_roadwork_L10_semseg.py
import os, glob
from pathlib import Path
from functools import partial
from detectron2.data import DatasetCatalog, MetadataCatalog

IGNORE_LABEL = 255
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

STUFF_CLASSES = [
    "road",
    "sidewalk",
    "barrier",
    "police vehicle",
    "work vehicle",
    "police officer",
    "worker",
    "cone",
    "arrow board",
    "ttc sign",
]

STUFF_COLORS = [
    (128,  64, 128),  # road
    (244,  35, 232),  # sidewalk
    (246, 116, 185),  # barrier
    (255,  68,  51),  # police vehicle
    (255, 104,  66),  # work vehicle
    (184, 107,  35),  # police officer
    (205, 135,  29),  # worker
    ( 30, 119, 179),  # cone
    (241,  71,  14),  # arrow board
    (254, 139,  32),  # ttc sign
]

def _match_image(images_root: Path, rel_stem: str):
    for ext in _IMG_EXTS:
        p = images_root / f"{rel_stem}{ext}"
        if p.is_file():
            return str(p)
    return None

def _roadwork_L10_loader(root: str, split: str):
    images_root = Path(root) / "images"
    labels_root = Path(root) / "gtFine_10" / split
    label_paths = sorted(glob.glob(str(labels_root / "**" / "*_trainIds.png"), recursive=True))
    dataset_dicts = []

    for lp in label_paths:
        lp = Path(lp)
        rel = lp.relative_to(labels_root)
        rel_stem = str(rel.with_suffix("")).replace("_trainIds", "")
        img_path = _match_image(images_root, rel_stem)
        if img_path is None:
            continue
        dataset_dicts.append({
            "file_name": img_path,
            "sem_seg_file_name": str(lp),
        })
    return dataset_dicts

def _roadwork_L10_loader_showflat(root: str):
    images_root = Path(root) / "ROADWork_show" / "images"
    labels_root = Path(root) / "ROADWork_show" / "gtFine_trainIds"

    label_paths = sorted(labels_root.glob("*_trainIds.png"))
    dataset_dicts = []

    for lp in label_paths:
        stem = lp.stem.replace("_trainIds", "")
        img_path = _match_image(images_root, stem)
        if img_path is None:
            continue
        dataset_dicts.append({
            "file_name": img_path,
            "sem_seg_file_name": str(lp),
        })
    return dataset_dicts

def register_roadwork_L10_semseg(name: str, root: str, split: str):
    DatasetCatalog.register(name, partial(_roadwork_L10_loader, root, split))
    meta = MetadataCatalog.get(name)
    meta.set(
        evaluator_type="sem_seg",
        ignore_label=IGNORE_LABEL,
        stuff_classes=STUFF_CLASSES,
        stuff_colors=STUFF_COLORS,
        image_root=os.path.join(root, "images"),
        sem_seg_root=os.path.join(root, "gtFine_10", split),
    )

def register_roadwork_L10_semseg_showflat(name: str, root: str):
    DatasetCatalog.register(name, partial(_roadwork_L10_loader_showflat, root))
    meta = MetadataCatalog.get(name)
    meta.set(
        evaluator_type="sem_seg",
        ignore_label=IGNORE_LABEL,
        stuff_classes=STUFF_CLASSES,
        stuff_colors=STUFF_COLORS,
        image_root=os.path.join(root, "ROADWork_show/images"),
        sem_seg_root=os.path.join(root, "ROADWork_show/gtFine_10"),
    )

def register_all_roadwork_10(root: str):
    register_roadwork_L10_semseg("roadwork10_semseg_val", root, "val")
    register_roadwork_L10_semseg_showflat("roadwork10_semseg_val_show", root)

register_all_roadwork_10("/data1/zd/ROADWork_Data")
