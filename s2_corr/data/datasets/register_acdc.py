# projects/CATSeg/data/register_gta5.py
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CITYSCAPES_19 = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle"
]

def register_ACDC(name, img_dir, label_dir, classes):
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

def register_all_ACDC(root="/data/zd/data/ACDC/"):
    img_dir = os.path.join(root, "rgb_anon/train/")
    label_dir = os.path.join(root, "gt/train")  
    register_ACDC("ACDC_train_seg", img_dir, label_dir, CITYSCAPES_19)

# 直接执行一次注册
register_all_ACDC()
