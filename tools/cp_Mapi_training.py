# -*- coding: utf-8 -*-
"""
从 Mapillary training 数据集中提取包含指定类的所有样本（图像+标签）
保存到目标 OV_31 目录。
"""

import os
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image

# -------------------- 配置部分 --------------------
SRC_LABEL_DIR = Path("/data/zd/data/Mapillary/mapillary/labels_TrainID30/")
SRC_IMAGE_DIR = Path("/data/zd/data/Mapillary/mapillary/training/images")
DST_LABEL_DIR = Path("/data/zd/data/Mapillary/mapillary/OV_30/labels")
DST_IMAGE_DIR = Path("/data/zd/data/Mapillary/mapillary/OV_30/images")

# Mapillary 31 类名称表
OPEN30 = [
    "road","sidewalk","building","wall","bridge","tunnel",
    "traffic sign","traffic light","pole","fence","sky",
    "vegetation","terrain","water","snow","sand",
    "person","rider","car","truck","bus","train",
    "bicycle","motorcycle","animal","signboard",
    "railway","boat","chair","trash can"
]
NAME2ID = {name.lower(): i for i, name in enumerate(OPEN30)}

# 需要筛选的类名  tunnel  boat animal chair
TARGET_CLASSES = [
     "railway",
     "chair",  "tunnel",  "sand"
]
TARGET_IDS = [NAME2ID[name.lower()] for name in TARGET_CLASSES]

# 支持的扩展名
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
LABEL_EXTS = (".png", ".tif", ".tiff")


# -------------------- 工具函数 --------------------
def find_image(base: str) -> str:
    """根据标签文件名查找对应图像"""
    for ext in IMAGE_EXTS:
        path = SRC_IMAGE_DIR / f"{base}{ext}"
        if path.exists():
            return str(path)
    return None


def find_label(base: str) -> str:
    """查找标签路径"""
    for ext in LABEL_EXTS:
        path = SRC_LABEL_DIR / f"{base}{ext}"
        if path.exists():
            return str(path)
    return None


def load_label_array(path: str) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[..., 0]
        elif arr.shape[2] == 3:
            # 如果三通道一致则取一通道
            if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
                arr = arr[..., 0]
            else:
                raise ValueError(f"标签文件 {path} 看起来是彩色标签，请先转为 TrainID。")
    return arr.astype(np.int32, copy=False)


def ensure_dir(path: Path):
    os.makedirs(path, exist_ok=True)


# -------------------- 主逻辑 --------------------
def main():
    ensure_dir(DST_LABEL_DIR)
    ensure_dir(DST_IMAGE_DIR)

    all_labels = [f for f in os.listdir(SRC_LABEL_DIR) if f.lower().endswith(LABEL_EXTS)]
    all_labels.sort()

    selected = []

    print(f"[INFO] 共检测 {len(all_labels)} 张标签，开始筛选包含目标类的样本...")
    for fname in tqdm(all_labels):
        base = os.path.splitext(fname)[0]
        label_path = SRC_LABEL_DIR / fname
        try:
            arr = load_label_array(label_path)
        except Exception as e:
            tqdm.write(f"[WARN] 跳过 {fname}: {e}")
            continue

        # 如果该标签中包含任何目标类ID，则拷贝
        if np.any(np.isin(arr, TARGET_IDS)):
            img_path = find_image(base)
            if img_path is None:
                tqdm.write(f"[WARN] 未找到图像: {base}")
                continue

            # 拷贝
            dst_lab = DST_LABEL_DIR / fname
            dst_img = DST_IMAGE_DIR / Path(img_path).name
            shutil.copy2(label_path, dst_lab)
            shutil.copy2(img_path, dst_img)
            selected.append(base)

    print(f"\n✅ 共提取 {len(selected)} 张样本")
    print(f"保存路径：\n  Images → {DST_IMAGE_DIR}\n  Labels → {DST_LABEL_DIR}")
    print(f"包含的目标类：{', '.join(TARGET_CLASSES)}")

# nohup python cp_training_OV31.py  >  logs/cp_training_OV31.log 2>&1 &


if __name__ == "__main__":
    main()
