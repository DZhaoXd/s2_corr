# projects/CATSeg/data/register_custom_street.py
import os, glob, json
from detectron2.data import DatasetCatalog, MetadataCatalog

def _custom_loader(img_dir, label_dir=None):
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    img_paths = sorted(
        p for p in glob.glob(os.path.join(img_dir, "**/*.*"), recursive=True)
        if os.path.splitext(p)[1].lower() in exts
    )
    ds = []
    for p in img_paths:
        rec = {"file_name": p}  
        if label_dir:
            rel = os.path.relpath(p, img_dir)
            sem = os.path.join(label_dir, os.path.splitext(rel)[0] + "_gt.png")
            if os.path.isfile(sem):
                rec["sem_seg_file_name"] = sem
        ds.append(rec)
    return ds

def register_custom_street(name, img_dir, label_dir=None, class_names=None):
    DatasetCatalog.register(name, lambda: _custom_loader(img_dir, label_dir))
    meta = MetadataCatalog.get(name)
    if class_names is None:
        with open("datasets/inpaint41_vocab.json", "r") as f:
            class_names = json.load(f)
    meta.set(
        stuff_classes=class_names,
        evaluator_type="sem_seg",
        ignore_label=255,
        image_root=img_dir,
        sem_seg_root=label_dir,
    )

def register_all_custom_street(root="/data/yrz/repos/sd_inpaint/Open_Class_BDD_v2_labeling_format"):
    img_dir = os.path.join(root, "images")
    label_dir = os.path.join(root, "labels")
    register_custom_street("bdd_inpaint41_test", img_dir, label_dir, class_names=None)

register_all_custom_street()
