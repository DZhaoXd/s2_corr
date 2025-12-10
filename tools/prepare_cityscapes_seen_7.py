from pathlib import Path
import numpy as np
from PIL import Image

# 映射表：Cityscapes trainId -> 连续 0~6
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
    for p in gt19_split.rglob("*_gtFine_labelTrainIds.png"):
        rel = p.relative_to(gt19_split)
        dst = gt7_split / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        m = np.array(Image.open(p), dtype=np.uint8)
        out = np.full_like(m, IGNORE_VALUE, dtype=np.uint8)
        # 遍历字典映射
        for old_id, new_id in REMAPPING.items():
            out[m == old_id] = new_id
        Image.fromarray(out).save(dst)

if __name__ == "__main__":
    root = ""
    process_split(f"{root}/gtFine_19/train", f"{root}/gtFine_7/train")
