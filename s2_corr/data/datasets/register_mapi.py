# projects/CATSeg/data/register_mapillary.py
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CITYSCAPES_19 = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle"
]

def register_Mapillary(name, img_dir, label_dir, classes=CITYSCAPES_19,
                       image_ext="jpg", gt_ext="png"):
    DatasetCatalog.register(
        name,
        lambda x=label_dir, y=img_dir, e_img=image_ext, e_gt=gt_ext: load_sem_seg(
            gt_root=x, image_root=y, gt_ext=e_gt, image_ext=e_img
        )
    )
    MetadataCatalog.get(name).set(
        stuff_classes=classes,
        evaluator_type="sem_seg",
        ignore_label=255,
        image_root=img_dir,
        sem_seg_root=label_dir,
    )

def register_all_Mapillary(root="/data/zd/data/Mapillary/mapillary/"):

    img_val = os.path.join(root, "val/images/")
    gt_val  = os.path.join(root, "val/labels_TrainIds/")
    register_Mapillary("mapillary_val_seg", img_val, gt_val)


register_all_Mapillary("/data/zd/data/Mapillary/mapillary/")
