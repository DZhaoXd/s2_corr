# projects/CATSeg/data/register_mapillary.py
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import json


def register_Mapillary(name, img_dir, label_dir, classes=None,
                       image_ext="jpg", gt_ext="png"):
    DatasetCatalog.register(
        name,
        lambda x=label_dir, y=img_dir, e_img=image_ext, e_gt=gt_ext: load_sem_seg(
            gt_root=x, image_root=y, gt_ext=e_gt, image_ext=e_img
        )
    )
    if classes is None:
        with open("datasets/mapi_30.json", "r") as f:
            classes = json.load(f)
    MetadataCatalog.get(name).set(
        stuff_classes=classes,
        evaluator_type="sem_seg",
        ignore_label=255,
        image_root=img_dir,
        sem_seg_root=label_dir,
    )

def register_all_Mapillary(root="/data/zd/data/Mapillary/"):
    img_val = os.path.join(root, "val/images/")
    gt_val  = os.path.join(root, "val/labels_TrainID30/")
    register_Mapillary("mapillary_val_seg_30", img_val, gt_val)

    img_train = os.path.join(root, "OV_30/images/")
    gt_train  = os.path.join(root, "OV_30/labels/")
    register_Mapillary("mapillary_train_val_seg_30", img_train, gt_train)

    img_train = os.path.join(root, "OV_30_show/images/")
    gt_train  = os.path.join(root, "OV_30_show/labels/")
    register_Mapillary("mapillary_show_seg_30", img_train, gt_train)


register_all_Mapillary("/data/zd/data/Mapillary/mapillary/")
