# projects/CATSeg/data/register_gta5.py
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import os, glob, json


CITYSCAPES_19 = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle"
]

def register_gta5(name, img_dir, label_dir, classes):
    DatasetCatalog.register(
        name,
        lambda x=label_dir, y=img_dir: load_sem_seg(
            gt_root=x, image_root=y,
            gt_ext="png", image_ext="png"
        )
    )
    MetadataCatalog.get(name).set(
        stuff_classes=classes,
        evaluator_type="sem_seg",
        ignore_label=255,
        image_root=img_dir,
        sem_seg_root=label_dir,
    )

def register_all_gta5(root="/data/zd/data/GTA5/GTAV/"):
    img_dir = os.path.join(root, "images")
    label_dir = os.path.join(root, "labels_19")
    register_gta5("gta5_train_seg", img_dir, label_dir, CITYSCAPES_19)

def register_all_gta5_seen7(root="/data/zd/data/GTA5/GTAV/"):
    img_dir = os.path.join(root, "images")
    label_dir = os.path.join(root, "labels_7")  
    with open("datasets/cityscapes_seen_7.json", "r") as f:
        class_names = json.load(f)
    register_gta5("gta5_seen7_train_seg", img_dir, label_dir, class_names)


# 直接执行一次注册
register_all_gta5()
register_all_gta5_seen7()
