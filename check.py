# import os
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
#
#
# def check_fmb_classes(label_dir):
#     """
#     校验 FMB 数据集中 Label 文件夹的类别数量。
#     参数:
#         label_dir (str): 标签路径，例如 /data/zd/dwp/dataset/FMB/train/Label/
#     """
#     assert os.path.exists(label_dir), f"路径不存在: {label_dir}"
#
#     all_classes = set()
#     image_list = [f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.tif'))]
#
#     print(f"共发现 {len(image_list)} 个标注文件，开始统计类别值...\n")
#
#     for img_name in tqdm(image_list):
#         path = os.path.join(label_dir, img_name)
#         img = np.array(Image.open(path))
#         unique_vals = np.unique(img)
#         all_classes.update(unique_vals.tolist())
#
#     all_classes = sorted(list(all_classes))
#
#     print("\n==== 统计结果 ====")
#     print(f"总类别数量: {len(all_classes)}")
#     print(f"类别值列表: {all_classes}")
#
#     if len(all_classes) == 14:
#         print("✅ 检测结果: 确实存在 14 个类别，与论文一致。")
#     else:
#         print("⚠️ 检测结果: 类别数量 != 14，请检查标注文件是否缺失或类别映射不同。")
#
#
# # 示例调用
# if __name__ == "__main__":
#     check_fmb_classes("/data/zd/dwp/dataset/FMB/train/Label/")


import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter


def analyze_fmb_classes(label_dir):
    """
    分析 FMB 数据集的类别分布，帮助确认是否存在背景类。
    """
    class_counter = Counter()
    image_list = [f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.tif'))]
    total_pixels = 0

    for img_name in tqdm(image_list, desc="统计中"):
        path = os.path.join(label_dir, img_name)
        img = np.array(Image.open(path))
        unique, counts = np.unique(img, return_counts=True)
        total_pixels += img.size
        for u, c in zip(unique, counts):
            class_counter[u] += c

    print("\n==== 类别统计结果 ====")
    print(f"检测到 {len(class_counter)} 个唯一类别ID\n")
    for cid, count in sorted(class_counter.items()):
        ratio = count / total_pixels * 100
        print(f"类别ID {cid:2d}: 像素数 {count:10d} ({ratio:.4f}%)")
    print("\n总像素数:", total_pixels)

    likely_bg = [cid for cid, count in class_counter.items() if count / total_pixels < 0.01]
    if 0 in class_counter and class_counter[0] / total_pixels > 0.2:
        print("\n✅ 可能的背景类: 0 (占比 {:.2f}%)".format(class_counter[0] / total_pixels * 100))
    if 255 in class_counter:
        print("⚠️ 注意: 检测到像素值 255，通常是 ignore / void 标签")
    if likely_bg:
        print(f"⚠️ 占比极低的类别ID: {likely_bg} (可能在训练中被忽略)")


# 调用
if __name__ == "__main__":
    analyze_fmb_classes("/data/zd/data/Mapillary/mapillary/val/labels_TrainID31")

